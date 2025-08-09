from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import pandas as pd
import os
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------
# App & DB setup
# -----------------
app = Flask(__name__)
app.secret_key = "msu_secret"  # replace in production

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///expense_tracker.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"

# -----------------
# Models
# -----------------
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

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# -----------------
# Constants for charts
# -----------------
BAR_CHART_PATH = "static/spending_by_category.png"
PIE_CHART_PATH = "static/spending_pie.png"

# -----------------
# Helpers
# -----------------
def _parse_money(value_str: str, field_label: str, max_abs: float = 1e7):
    """Validate and parse a numeric currency-like field (>=0, finite, reasonable)."""
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

# -----------------
# Auth routes (minimal templates required: templates/login.html, templates/register.html)
# -----------------
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

# -----------------
# Main app
# -----------------
@app.route("/", methods=["GET", "POST"])
@login_required
def index():
    # Query params for edit states (use DB ids now)
    tx_edit_id_q = request.args.get("edit")       # transaction id
    goal_edit_id_q = request.args.get("goal_edit") # goal id

    # ----- HANDLE TRANSACTION CREATE/UPDATE (POST /) -----
    if request.method == "POST" and "amount" in request.form and "date" in request.form:
        amount_valid = _parse_money(request.form.get("amount"), "Amount")
        if amount_valid is None:
            tx_edit_ctx = request.form.get("edit_index")
            if tx_edit_ctx:
                return redirect(url_for("index", edit=tx_edit_ctx))
            return redirect(url_for("index"))

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
                flash("Invalid transaction id.")
        else:
            tx = Transaction(
                user_id=current_user.id,
                date=request.form["date"],
                description=request.form["description"].strip(),
                category=request.form["category"].strip(),
                amount=amount_valid,
            )
            db.session.add(tx)
            db.session.commit()
            flash("Transaction added successfully!")
        return redirect(url_for("index"))

    # ----- Pull user data from DB -----
    tx_rows = Transaction.query.filter_by(user_id=current_user.id).order_by(Transaction.date.desc(), Transaction.id.desc()).all()
    goal_rows = Goal.query.filter_by(user_id=current_user.id).all()

    # Convert to DataFrames for existing pandas/chart logic
    df = pd.DataFrame([
        {"Id": t.id, "Date": t.date, "Description": t.description, "Category": t.category, "Amount": t.amount}
        for t in tx_rows
    ])
    goals_df = pd.DataFrame([
        {"GoalId": g.id, "Month": g.month, "Category": g.category, "Goal": g.goal}
        for g in goal_rows
    ])

    # ---- Coerce dtypes safely ----
    if not df.empty:
        df["Amount"] = pd.to_numeric(df.get("Amount", pd.Series()), errors="coerce")
    if not goals_df.empty:
        goals_df["Goal"] = pd.to_numeric(goals_df.get("Goal", pd.Series()), errors="coerce")

    # ----- Charts (only if there is data) -----
    if not df.empty and "Category" in df.columns:
        category_totals = df.groupby("Category")["Amount"].sum().sort_values(ascending=False)
        if len(category_totals) > 0 and (category_totals.sum() or 0) > 0:
            msu_colors = ["#18453B", "#4E8B6C", "#CCCCCC", "#FFFFFF"]
            fallback_colors = plt.cm.tab20.colors
            color_palette = msu_colors + list(fallback_colors)
            color_list = color_palette[:len(category_totals)]

            plt.figure(figsize=(8, 5))
            category_totals.plot(kind="bar", color=color_list)
            plt.title("Spending by Category")
            plt.ylabel("Total ($)")
            plt.tight_layout()
            plt.savefig(BAR_CHART_PATH)
            plt.close()

            plt.figure(figsize=(6, 6))
            plt.pie(category_totals, labels=category_totals.index, autopct="%1.1f%%",
                    startangle=140, colors=color_list)
            plt.title("Spending Distribution by Category")
            plt.axis("equal")
            plt.tight_layout()
            plt.savefig(PIE_CHART_PATH)
            plt.close()
        else:
            for p in (BAR_CHART_PATH, PIE_CHART_PATH):
                if os.path.exists(p):
                    try: os.remove(p)
                    except OSError: pass
    else:
        for p in (BAR_CHART_PATH, PIE_CHART_PATH):
            if os.path.exists(p):
                try: os.remove(p)
                except OSError: pass

    # ----- Comparison table (preserve GoalId for actions) -----
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

    # ----- Edit states -----
    tx_edit_data = None
    if tx_edit_id_q and tx_edit_id_q.isdigit():
        tx = Transaction.query.filter_by(id=int(tx_edit_id_q), user_id=current_user.id).first()
        if tx:
            tx_edit_data = {"Id": tx.id, "Date": tx.date, "Description": tx.description, "Category": tx.category, "Amount": tx.amount}

    goal_edit_data = None
    if goal_edit_id_q and goal_edit_id_q.isdigit():
        g = Goal.query.filter_by(id=int(goal_edit_id_q), user_id=current_user.id).first()
        if g:
            goal_edit_data = {"GoalId": g.id, "Month": g.month, "Category": g.category, "Goal": g.goal}

    transactions = [
        {"Id": t.id, "Date": t.date, "Description": t.description, "Category": t.category, "Amount": t.amount}
        for t in tx_rows
    ]
    has_transactions = len(transactions) > 0

    return render_template(
        "index.html",
        transactions=transactions,
        comparisons=comparison.to_dict(orient="records"),
        edit_data=tx_edit_data,
        goal_edit_data=goal_edit_data,
        has_transactions=has_transactions,
    )

# ----- Mutations (Transactions) -----
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

# ----- Mutations (Goals) -----
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

    if goal_edit_id:  # update existing by id
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

    # upsert by (month, category)
    g = Goal.query.filter_by(user_id=current_user.id, month=goal_month, category=goal_category).first()
    if g:
        g.goal = goal_amount_valid
    else:
        g = Goal(user_id=current_user.id, month=goal_month, category=goal_category, goal=goal_amount_valid)
        db.session.add(g)
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

# -----------------
# One-time DB init
# -----------------
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)