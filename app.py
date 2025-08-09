import os
import math
import pandas as pd
from datetime import datetime, timedelta

from flask import (
    Flask, render_template, request, redirect, url_for, flash, session
)
from flask_sqlalchemy import SQLAlchemy
from flask_login import (
    LoginManager, login_user, login_required, logout_user, current_user, UserMixin
)
from werkzeug.security import generate_password_hash, check_password_hash

# ---- Load .env early so env vars are available ----
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Optional OpenAI import (we’ll detect SDK version at runtime)
try:
    import openai
except Exception:
    openai = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# App / Config
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "msu_secret")

# DB config: keep SQLite locally; if you later set DATABASE_URL, it will use that
db_url = os.getenv("DATABASE_URL", "sqlite:///expense_tracker.db")
if db_url.startswith("postgres://"):
    db_url = db_url.replace("postgres://", "postgresql://", 1)
app.config["SQLALCHEMY_DATABASE_URI"] = db_url
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# Chart output paths (can be set to /tmp by CHART_DIR env if desired)
CHART_DIR = os.getenv("CHART_DIR", "static")
os.makedirs(CHART_DIR, exist_ok=True)
BAR_CHART_PATH = os.path.join(CHART_DIR, "spending_by_category.png")
PIE_CHART_PATH = os.path.join(CHART_DIR, "spending_pie.png")


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

class Transaction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    date = db.Column(db.String(10), nullable=False)  # YYYY-MM-DD
    description = db.Column(db.String(255), nullable=False)
    category = db.Column(db.String(120), nullable=False)
    amount = db.Column(db.Float, nullable=False)

class Goal(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, index=True)
    month = db.Column(db.String(7), nullable=False)  # YYYY-MM
    category = db.Column(db.String(120), nullable=False)
    goal = db.Column(db.Float, nullable=False)

# Simple cache of generated insights (per user, per month)
class AIInsight(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, index=True, nullable=False)
    month = db.Column(db.String(7), index=True, nullable=False)  # YYYY-MM
    text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


@login_manager.user_loader
def load_user(user_id):
    # SQLAlchemy 2.0 style: avoids legacy warning
    return db.session.get(User, int(user_id))


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _parse_money(value_str: str, field_label: str, max_abs: float = 1e7):
    """Validate and parse a currency-like field (>=0, finite, <= max_abs)."""
    if value_str is None or str(value_str).strip() == "":
        flash(f"{field_label} is required.")
        return None
    try:
        val = float(value_str)
    except ValueError:
        flash(f"{field_label} must be a number.")
        return None
    if not math.isfinite(val):
        flash(f"{field_label} must be a finite number.")
        return None
    if val < 0:
        flash(f"{field_label} cannot be negative.")
        return None
    if abs(val) > max_abs:
        flash(f"{field_label} is unreasonably large (>{int(max_abs):,}).")
        return None
    return round(val, 2)


def _df_for_current_user():
    tx_rows = Transaction.query.filter_by(user_id=current_user.id).all()
    df = pd.DataFrame([
        {"Id": t.id, "Date": t.date, "Description": t.description,
         "Category": t.category, "Amount": t.amount}
        for t in tx_rows
    ])
    if not df.empty:
        df["Amount"] = pd.to_numeric(df.get("Amount", pd.Series()), errors="coerce").fillna(0.0)
    return df


def _build_spending_summary(df: pd.DataFrame):
    """Return (by_category_df, total_amount)."""
    if df.empty:
        return pd.DataFrame(columns=["Category", "Amount"]), 0.0
    by_cat = (
        df.groupby("Category", dropna=False)["Amount"]
          .sum().reset_index()
          .sort_values("Amount", ascending=False)
    )
    total = float(by_cat["Amount"].sum()) if not by_cat.empty else 0.0
    return by_cat, total


# ---- Version-compatible OpenAI call with quota-aware errors ----
def _call_openai(prompt: str) -> str:
    """
    Works with both OpenAI SDK v1.x and v0.x.
    Raises RuntimeError('AI quota exceeded') on insufficient quota,
    or re-raises other exceptions for caller to handle.
    """
    import sys, traceback
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    if openai is None or not api_key:
        raise RuntimeError("OpenAI SDK missing or OPENAI_API_KEY not set.")

    try:
        # New SDK (v1.x)
        if hasattr(openai, "OpenAI"):
            client = openai.OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
            )
            return resp.choices[0].message.content.strip()

        # Old SDK (v0.x)
        openai.api_key = api_key
        resp = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return resp.choices[0].message["content"].strip()

    except Exception as e:
        err = str(e)
        print("OpenAI error:", repr(e))
        import traceback as tb; tb.print_exc()
        if "insufficient_quota" in err or "exceeded your current quota" in err:
            raise RuntimeError("AI quota exceeded")
        raise


def _generate_ai_insights(df: pd.DataFrame) -> str:
    """
    Generate brief spending insights.
    Uses OpenAI if OPENAI_API_KEY is set and call succeeds; otherwise deterministic tips.
    """
    if df.empty or df["Amount"].sum() <= 0:
        return "No spending data yet—add a few transactions to see insights."

    by_cat, total = _build_spending_summary(df)
    lines = [f"{row.Category}: ${row.Amount:.2f}" for _, row in by_cat.iterrows()]
    summary_text = "\n".join(lines[:12])

    prompt = f"""
You are a concise personal finance assistant. Based on the user's spending by category (below),
provide 3 brief, actionable insights to help them save money or stick to goals.
Be specific and numerical where possible.

Spending by category:
{summary_text}
Total this period: ${total:.2f}

Return a short bullet list (max 3 bullets).
""".strip()

    try:
        return _call_openai(prompt)
    except RuntimeError as e:
        if "quota exceeded" in str(e).lower():
            flash("AI quota exceeded — showing basic insights instead.")
        else:
            flash("AI service unavailable — showing basic insights instead.")
    except Exception:
        flash("AI service unavailable — showing basic insights instead.")

    # Deterministic fallback
    top = by_cat.iloc[0] if not by_cat.empty else None
    tip1 = (f"Top category: {top.Category} at ${top.Amount:.2f}. "
            f"Consider setting a tighter goal.") if top is not None else "Review your top categories to identify quick savings."
    avg = total / max(len(by_cat), 1)
    tip2 = f"Average per category: ${avg:.2f}. Target categories above average for reductions."
    try:
        df["Month"] = pd.to_datetime(df["Date"], errors="coerce").dt.to_period("M").astype(str)
        mt = df.groupby(["Month"])["Amount"].sum().reset_index()
        last_two = sorted(mt["Month"].unique())[-2:]
        if len(last_two) == 2:
            m1, m2 = last_two
            s1 = float(mt[mt["Month"] == m1]["Amount"].sum())
            s2 = float(mt[mt["Month"] == m2]["Amount"].sum())
            delta = s2 - s1
            pct = (delta / s1 * 100) if s1 else 0.0
            trend = "up" if delta > 0 else ("down" if delta < 0 else "flat")
            tip3 = f"Spending is {trend} {abs(pct):.1f}% vs {m1}."
        else:
            tip3 = "Add another month of data to see a trend."
    except Exception:
        tip3 = "Add accurate dates to enable trend analysis."
    return "\n".join([f"• {tip1}", f"• {tip2}", f"• {tip3}"])


def _render_charts(df: pd.DataFrame):
    """Create charts only when there is data; remove old images otherwise."""
    has_transactions = not df.empty and "Category" in df.columns
    if has_transactions:
        category_totals = df.groupby("Category")["Amount"].sum().sort_values(ascending=False)
        if len(category_totals) > 0 and float(category_totals.sum()) > 0:
            msu_colors = ["#18453B", "#4E8B6C", "#CCCCCC", "#FFFFFF"]
            fallback_colors = plt.cm.tab20.colors
            color_palette = msu_colors + list(fallback_colors)
            color_list = color_palette[:len(category_totals)]

            # Bar
            plt.figure(figsize=(8, 5))
            category_totals.plot(kind="bar", color=color_list)
            plt.title("Spending by Category")
            plt.ylabel("Total ($)")
            plt.tight_layout()
            plt.savefig(BAR_CHART_PATH)
            plt.close()

            # Pie
            plt.figure(figsize=(6, 6))
            plt.pie(category_totals, labels=category_totals.index, autopct="%1.1f%%",
                    startangle=140, colors=color_list)
            plt.title("Spending Distribution by Category")
            plt.axis("equal")
            plt.tight_layout()
            plt.savefig(PIE_CHART_PATH)
            plt.close()
            return True

    # No data → clean up stale images
    for p in (BAR_CHART_PATH, PIE_CHART_PATH):
        if os.path.exists(p):
            try: os.remove(p)
            except OSError: pass
    return False


# -----------------------------------------------------------------------------
# Auth
# -----------------------------------------------------------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        pw = request.form["password"]
        if not email or not pw:
            flash("Email and password are required.")
            return redirect(url_for("register"))
        if User.query.filter_by(email=email).first():
            flash("Email already registered.")
            return redirect(url_for("register"))
        user = User(email=email, password_hash=generate_password_hash(pw))
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for("index"))
    return render_template("login.html", mode="register")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        pw = request.form["password"]
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, pw):
            login_user(user)
            return redirect(url_for("index"))
        flash("Invalid credentials.")
    return render_template("login.html", mode="login")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))


# -----------------------------------------------------------------------------
# Main + Insights
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    # Create/Update transaction
    if request.method == "POST" and "amount" in request.form and "date" in request.form:
        amount_valid = _parse_money(request.form.get("amount"), "Amount")
        if amount_valid is None:
            edit_ctx = request.form.get("edit_index")
            return redirect(url_for("index", edit=edit_ctx) if edit_ctx else url_for("index"))

        edit_id = request.form.get("edit_index")
        if edit_id:
            tx = Transaction.query.filter_by(id=int(edit_id), user_id=current_user.id).first()
            if tx:
                tx.date = request.form["date"]
                tx.description = request.form["description"].strip()
                tx.category = request.form["category"].strip()
                tx.amount = amount_valid
                db.session.commit()
                flash("Transaction updated successfully!")
        else:
            db.session.add(Transaction(
                user_id=current_user.id,
                date=request.form["date"],
                description=request.form["description"].strip(),
                category=request.form["category"].strip(),
                amount=amount_valid,
            ))
            db.session.commit()
            flash("Transaction added successfully!")
        return redirect(url_for("index"))

    # Load data
    tx_rows = Transaction.query.filter_by(user_id=current_user.id) \
                               .order_by(Transaction.date.desc(), Transaction.id.desc()).all()
    goal_rows = Goal.query.filter_by(user_id=current_user.id).all()

    df = pd.DataFrame([
        {"Id": t.id, "Date": t.date, "Description": t.description,
         "Category": t.category, "Amount": t.amount}
        for t in tx_rows
    ])
    goals_df = pd.DataFrame([
        {"GoalId": g.id, "Month": g.month, "Category": g.category, "Goal": g.goal}
        for g in goal_rows
    ])

    # Charts
    has_transactions = _render_charts(df)

    # Comparison table
    goals_reset = goals_df.reset_index(drop=True)
    if not df.empty:
        df["Month"] = pd.to_datetime(df.get("Date"), errors="coerce").dt.to_period("M").astype(str)
        monthly_totals = (
            df.dropna(subset=["Month"])
              .groupby(["Month", "Category"], dropna=False)["Amount"].sum()
              .reset_index()
        )
    else:
        monthly_totals = pd.DataFrame(columns=["Month", "Category", "Amount"])

    comparison = pd.merge(
        goals_reset, monthly_totals, how="left", on=["Month", "Category"]
    ) if not goals_reset.empty else pd.DataFrame(columns=["GoalId","Month","Category","Goal","Amount"])

    if not comparison.empty:
        comparison["Amount"] = pd.to_numeric(comparison.get("Amount"), errors="coerce").fillna(0).round(2)
        comparison["Goal"]   = pd.to_numeric(comparison.get("Goal"),   errors="coerce").fillna(0).round(2)
        comparison["Remaining"] = (comparison["Goal"] - comparison["Amount"]).round(2)
        comparison["Status"] = comparison["Remaining"].apply(lambda x: "Under Budget" if x >= 0 else "Over Budget")

    insights_text = session.pop("insights_text", None)

    return render_template(
        "index.html",
        transactions=df.to_dict(orient="records") if not df.empty else [],
        comparisons=comparison.to_dict(orient="records"),
        edit_data=None,
        goal_edit_data=None,
        has_transactions=has_transactions,
        insights_text=insights_text,
    )


@app.route("/insights", methods=["POST"])
@login_required
def insights():
    df = _df_for_current_user()
    if df.empty:
        session["insights_text"] = "No spending data yet—add a few transactions to see insights."
        return redirect(url_for("index"))

    # Determine the latest month present in user's data for caching
    month = pd.to_datetime(df["Date"], errors="coerce").dt.to_period("M").astype(str).max()
    force = request.form.get("force") == "1"  # <-- allow bypassing cache

    if month and not force:
        cached = (AIInsight.query
                  .filter_by(user_id=current_user.id, month=month)
                  .order_by(AIInsight.created_at.desc())
                  .first())
        if cached and (datetime.utcnow() - cached.created_at) < timedelta(days=1):
            session["insights_text"] = cached.text
            return redirect(url_for("index"))

    # Generate fresh insights (AI or fallback)
    text = _generate_ai_insights(df)

    # Store in cache (best-effort)
    try:
        if month:
            db.session.add(AIInsight(user_id=current_user.id, month=month, text=text))
            db.session.commit()
    except Exception:
        db.session.rollback()

    session["insights_text"] = text
    return redirect(url_for("index"))


# -----------------------------------------------------------------------------
# Transactions & Goals
# -----------------------------------------------------------------------------
@app.route("/delete", methods=["POST"])
@login_required
def delete():
    tx_id = int(request.form["id"])
    tx = Transaction.query.filter_by(id=tx_id, user_id=current_user.id).first()
    if tx:
        db.session.delete(tx)
        db.session.commit()
        flash("Transaction deleted successfully!")
    else:
        flash("Invalid transaction id.")
    return redirect(url_for("index"))


@app.route("/set-goal", methods=["POST"])
@login_required
def set_goal():
    goal_month = request.form["goal_month"]
    goal_category = request.form["goal_category"].strip()
    goal_amount_valid = _parse_money(request.form.get("goal_amount"), "Goal Amount")
    goal_edit_id = request.form.get("goal_edit_index")

    if goal_amount_valid is None:
        if goal_edit_id:
            return redirect(url_for("index", goal_edit=goal_edit_id))
        return redirect(url_for("index"))

    if goal_edit_id:
        g = Goal.query.filter_by(id=int(goal_edit_id), user_id=current_user.id).first()
        if g:
            g.month = goal_month
            g.category = goal_category
            g.goal = goal_amount_valid
            db.session.commit()
            flash("Goal updated successfully!")
        else:
            flash("Invalid goal id.")
        return redirect(url_for("index"))

    g = Goal.query.filter_by(user_id=current_user.id, month=goal_month, category=goal_category).first()
    if g:
        g.goal = goal_amount_valid
    else:
        db.session.add(Goal(user_id=current_user.id, month=goal_month, category=goal_category, goal=goal_amount_valid))
    db.session.commit()
    flash("Goal saved successfully!")
    return redirect(url_for("index"))


@app.route("/delete-goal", methods=["POST"])
@login_required
def delete_goal():
    goal_id = int(request.form["goal_id"])
    g = Goal.query.filter_by(id=goal_id, user_id=current_user.id).first()
    if g:
        db.session.delete(g)
        db.session.commit()
        flash("Goal deleted successfully!")
    else:
        flash("Invalid goal id.")
    return redirect(url_for("index"))


# -----------------------------------------------------------------------------
# Tiny health/debug routes (safe to keep during dev)
# -----------------------------------------------------------------------------
@app.route("/ai-health")
def ai_health():
    return f"Has key: {bool(os.getenv('OPENAI_API_KEY'))}"

@app.route("/insights-json")
@login_required
def insights_json():
    df = _df_for_current_user()
    try:
        text = _generate_ai_insights(df)
    except Exception as e:
        text = f"ERROR: {e}"
    return {"insights": text}


# -----------------------------------------------------------------------------
# App entry
# -----------------------------------------------------------------------------
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)
    