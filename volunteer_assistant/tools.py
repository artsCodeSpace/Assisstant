import math

from langchain_core.tools import tool


SKILL_LABELS = {
    "communication": "Communication",
    "patience": "Patience",
    "teamwork": "Teamwork",
    "organization": "Organization",
    "computer_basics": "Computer basics",
}


ROLES = {
    "Reading Buddy": {
        "required_skills": {"communication", "patience"},
        "minimum_hours": 2,
    },
    "Event Support": {
        "required_skills": {"teamwork", "organization"},
        "minimum_hours": 4,
    },
    "Digital Help": {
        "required_skills": {"computer_basics", "communication"},
        "minimum_hours": 3,
    },
}


@tool
def match_volunteer_roles(
    skills: list[str],
    available_hours_per_week: float,
) -> dict:
    """Match declared skills and weekly availability to volunteer roles.

    Each role is assessed independently. Return matching roles and
    the missing skills and additional hours for every other role.
    This tool does not register volunteers or allocate shifts.
    """

    if (
        not math.isfinite(available_hours_per_week)
        or available_hours_per_week < 0
    ):
        raise ValueError(
            "Available hours must be a finite, non-negative number."
        )

    # A set automatically removes duplicate skills.
    supplied_skills = set(skills)

    unknown_skills = supplied_skills - set(SKILL_LABELS)

    if unknown_skills:
        raise ValueError(
            "Unknown skill codes: " + ", ".join(sorted(unknown_skills))
        )

    matching_roles = []
    other_roles = []

    # Alphabetical iteration also keeps the output predictable.
    for role_name in sorted(ROLES):
        role = ROLES[role_name]

        missing_skills = sorted(
            role["required_skills"] - supplied_skills
        )

        additional_hours = max(
            0.0,
            role["minimum_hours"] - available_hours_per_week,
        )

        if not missing_skills and additional_hours == 0:
            matching_roles.append(role_name)
        else:
            other_roles.append({
                "role": role_name,
                "missing_skills": missing_skills,
                "additional_hours_needed": additional_hours,
            })

    return {
        "matching_roles": matching_roles,
        "other_roles": other_roles,
    }