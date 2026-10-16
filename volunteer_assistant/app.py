import math

from flask import Flask, render_template, request

from rag import graph
from tools import SKILL_LABELS


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    question = ""
    answer = None
    sources = []

    selected_skills = []
    hours_value = ""
    tool_result = None

    error = None

    if request.method == "POST":
        action = request.form.get("action", "")

        # Preserve form values after submission.
        question = request.form.get("question", "").strip()
        selected_skills = request.form.getlist("skills")
        hours_value = request.form.get("available_hours", "").strip()

        try:
            if action == "ask":
                if not question:
                    raise ValueError("Please enter a question.")

                result = graph.invoke({
                    "action": "ask",
                    "question": question,
                })

                answer = result["answer"]
                sources = result["sources"]

            elif action == "match_volunteer_roles":
                try:
                    available_hours = float(hours_value)
                except ValueError:
                    raise ValueError(
                        "Please enter a valid number of available hours."
                    ) from None

                if (
                    not math.isfinite(available_hours)
                    or available_hours < 0
                ):
                    raise ValueError(
                        "Available hours must be a finite, "
                        "non-negative number."
                    )

                unknown_skills = set(selected_skills) - set(SKILL_LABELS)

                if unknown_skills:
                    raise ValueError("Please select valid skills.")

                result = graph.invoke({
                    "action": "match_volunteer_roles",
                    "skills": selected_skills,
                    "available_hours_per_week": available_hours,
                })

                tool_result = result["tool_result"]

            else:
                raise ValueError("Please select a supported action.")

        except ValueError as exc:
            error = str(exc)

        except Exception:
            app.logger.exception("Assistant request failed")
            error = (
                "The request could not be completed. "
                "Check the terminal for details. For programme questions, "
                "check ingestion, internet access, and Groq configuration."
            )

    return render_template(
        "index.html",
        skill_labels=SKILL_LABELS,
        question=question,
        answer=answer,
        sources=sources,
        selected_skills=selected_skills,
        hours_value=hours_value,
        tool_result=tool_result,
        error=error,
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False,
    )