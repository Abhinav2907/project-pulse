import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_project_data(project_type="normal", seed=None):
    """
    Generate Jira data with different characteristics
    
    project_type options:
    - "normal": Balanced, healthy project
    - "bug_heavy": High bug count, quality issues
    - "fast_velocity": High velocity but accumulating tech debt
    - "slow_steady": Low velocity but high quality
    - "integration_hell": Lots of blocked tickets
    """
    
    if seed:
        np.random.seed(seed)
        random.seed(seed)
    
    data = []
    issue_id = 1000
    
    # Sprint configuration
    sprints = [f"Sprint {i}" for i in range(20, 26)]
    sprint_dates = {
        "Sprint 20": (datetime(2024, 10, 1), datetime(2024, 10, 14)),
        "Sprint 21": (datetime(2024, 10, 15), datetime(2024, 10, 28)),
        "Sprint 22": (datetime(2024, 10, 29), datetime(2024, 11, 11)),
        "Sprint 23": (datetime(2024, 11, 12), datetime(2024, 11, 25)),
        "Sprint 24": (datetime(2024, 11, 26), datetime(2024, 12, 9)),
        "Sprint 25": (datetime(2024, 12, 10), datetime(2024, 12, 23)),
    }
    
    issue_types = ["Story", "Bug", "Task", "Epic", "Tech Debt"]
    priorities = ["Critical", "High", "Medium", "Low"]
    statuses = ["To Do", "In Progress", "In Review", "Done", "Blocked"]
    components = ["Backend", "Frontend", "API", "Database", "Infrastructure", "Mobile"]
    
    # Define project profiles
    if project_type == "bug_heavy":
        # High bug count, quality issues
        sprint_profiles = {
            "Sprint 20": {"Story": 12, "Bug": 15, "Task": 4, "Tech Debt": 2, "Epic": 1},
            "Sprint 21": {"Story": 10, "Bug": 18, "Task": 3, "Tech Debt": 1, "Epic": 1},
            "Sprint 22": {"Story": 11, "Bug": 20, "Task": 3, "Tech Debt": 2, "Epic": 1},
            "Sprint 23": {"Story": 9, "Bug": 22, "Task": 2, "Tech Debt": 1, "Epic": 1},
            "Sprint 24": {"Story": 10, "Bug": 19, "Task": 4, "Tech Debt": 2, "Epic": 1},
            "Sprint 25": {"Story": 8, "Bug": 21, "Task": 3, "Tech Debt": 3, "Epic": 1},
        }
    elif project_type == "fast_velocity":
        # High velocity but tech debt growing
        sprint_profiles = {
            "Sprint 20": {"Story": 22, "Bug": 6, "Task": 8, "Tech Debt": 2, "Epic": 2},
            "Sprint 21": {"Story": 24, "Bug": 7, "Task": 7, "Tech Debt": 3, "Epic": 2},
            "Sprint 22": {"Story": 26, "Bug": 8, "Task": 6, "Tech Debt": 4, "Epic": 1},
            "Sprint 23": {"Story": 25, "Bug": 9, "Task": 5, "Tech Debt": 5, "Epic": 2},
            "Sprint 24": {"Story": 23, "Bug": 10, "Task": 6, "Tech Debt": 6, "Epic": 1},
            "Sprint 25": {"Story": 20, "Bug": 11, "Task": 5, "Tech Debt": 7, "Epic": 2},
        }
    elif project_type == "slow_steady":
        # Low velocity but high quality
        sprint_profiles = {
            "Sprint 20": {"Story": 10, "Bug": 3, "Task": 5, "Tech Debt": 1, "Epic": 1},
            "Sprint 21": {"Story": 11, "Bug": 2, "Task": 6, "Tech Debt": 1, "Epic": 1},
            "Sprint 22": {"Story": 10, "Bug": 3, "Task": 5, "Tech Debt": 2, "Epic": 1},
            "Sprint 23": {"Story": 12, "Bug": 2, "Task": 4, "Tech Debt": 1, "Epic": 1},
            "Sprint 24": {"Story": 11, "Bug": 3, "Task": 5, "Tech Debt": 1, "Epic": 1},
            "Sprint 25": {"Story": 10, "Bug": 2, "Task": 6, "Tech Debt": 2, "Epic": 1},
        }
    elif project_type == "integration_hell":
        # Lots of blocked tickets
        sprint_profiles = {
            "Sprint 20": {"Story": 15, "Bug": 8, "Task": 5, "Tech Debt": 3, "Epic": 2},
            "Sprint 21": {"Story": 16, "Bug": 10, "Task": 6, "Tech Debt": 2, "Epic": 1},
            "Sprint 22": {"Story": 14, "Bug": 12, "Task": 4, "Tech Debt": 2, "Epic": 2},
            "Sprint 23": {"Story": 13, "Bug": 11, "Task": 5, "Tech Debt": 1, "Epic": 1},
            "Sprint 24": {"Story": 15, "Bug": 9, "Task": 6, "Tech Debt": 2, "Epic": 1},
            "Sprint 25": {"Story": 14, "Bug": 10, "Task": 7, "Tech Debt": 4, "Epic": 2},
        }
    else:  # normal/balanced
        sprint_profiles = {
            "Sprint 20": {"Story": 15, "Bug": 8, "Task": 5, "Tech Debt": 3, "Epic": 2},
            "Sprint 21": {"Story": 16, "Bug": 7, "Task": 6, "Tech Debt": 2, "Epic": 1},
            "Sprint 22": {"Story": 18, "Bug": 9, "Task": 4, "Tech Debt": 2, "Epic": 2},
            "Sprint 23": {"Story": 20, "Bug": 12, "Task": 5, "Tech Debt": 1, "Epic": 1},
            "Sprint 24": {"Story": 17, "Bug": 11, "Task": 6, "Tech Debt": 2, "Epic": 1},
            "Sprint 25": {"Story": 14, "Bug": 10, "Task": 7, "Tech Debt": 4, "Epic": 2},
        }
    
    for sprint, counts in sprint_profiles.items():
        sprint_start, sprint_end = sprint_dates[sprint]
        
        for issue_type, count in counts.items():
            for _ in range(count):
                days_offset = random.randint(0, 7)
                created = sprint_start + timedelta(days=days_offset)
                
                is_current_sprint = (sprint == "Sprint 25")
                if is_current_sprint:
                    status = random.choices(
                        statuses, 
                        weights=[0.15, 0.30, 0.25, 0.25, 0.05]
                    )[0]
                else:
                    status = random.choices(
                        statuses,
                        weights=[0.05, 0.05, 0.05, 0.80, 0.05]
                    )[0]
                
                # Adjust blocked probability based on project type
                if project_type == "integration_hell" and random.random() > 0.7:
                    status = "Blocked"
                
                if issue_type == "Epic":
                    story_points = random.choice([13, 21, 34])
                elif issue_type == "Story":
                    story_points = random.choice([1, 2, 3, 5, 8])
                elif issue_type == "Tech Debt":
                    story_points = random.choice([3, 5, 8])
                elif issue_type == "Bug":
                    story_points = random.choice([1, 2, 3])
                else:
                    story_points = random.choice([1, 2, 3])
                
                if issue_type == "Bug":
                    priority = random.choices(
                        priorities,
                        weights=[0.15, 0.35, 0.40, 0.10]
                    )[0]
                else:
                    priority = random.choices(
                        priorities,
                        weights=[0.05, 0.25, 0.50, 0.20]
                    )[0]
                
                if status == "Done":
                    resolution_days = random.randint(2, 10)
                    resolved = created + timedelta(days=resolution_days)
                else:
                    resolved = None
                
                component = random.choice(components)
                assignees = ["Alice Chen", "Bob Smith", "Carol Davis", "David Kim", 
                           "Emma Wilson", "Frank Zhang", "Grace Lee"]
                assignee = random.choice(assignees)
                
                labels = []
                if issue_type == "Bug" and priority in ["Critical", "High"]:
                    labels.append("production-bug")
                if issue_type == "Tech Debt":
                    labels.append("tech-debt")
                if random.random() > 0.7:
                    labels.append("customer-request")
                
                data.append({
                    "Issue Key": f"PROJ-{issue_id}",
                    "Issue Type": issue_type,
                    "Summary": f"{issue_type}: {component} enhancement",
                    "Status": status,
                    "Priority": priority,
                    "Sprint": sprint,
                    "Story Points": story_points,
                    "Assignee": assignee,
                    "Created": created.strftime("%Y-%m-%d"),
                    "Resolved": resolved.strftime("%Y-%m-%d") if resolved else "",
                    "Component": component,
                    "Labels": ";".join(labels) if labels else "",
                    "Original Estimate (hours)": story_points * 4 if story_points else 0,
                    "Time Spent (hours)": story_points * 4.5 if status == "Done" else story_points * 2,
                })
                
                issue_id += 1
    
    return pd.DataFrame(data)

# Generate multiple project scenarios
if __name__ == "__main__":
    print("=" * 60)
    print("Generating Test Projects for RAG Demo")
    print("=" * 60)
    print()
    
    projects = {
        "project_alpha_normal.csv": ("normal", "Balanced project"),
        "project_beta_buggy.csv": ("bug_heavy", "Quality issues, high bug rate"),
        "project_gamma_fast.csv": ("fast_velocity", "High velocity, tech debt growing"),
        "project_delta_steady.csv": ("slow_steady", "Low velocity, high quality"),
        "project_epsilon_blocked.csv": ("integration_hell", "Integration issues, blocked tickets")
    }
    
    for filename, (project_type, description) in projects.items():
        df = generate_project_data(project_type=project_type)
        df.to_csv(filename, index=False)
        
        bug_rate = len(df[df['Issue Type'] == 'Bug']) / len(df) * 100
        velocity = df.groupby('Sprint')['Story Points'].sum().mean()
        
        print(f"✅ {filename}")
        print(f"   Description: {description}")
        print(f"   Total Issues: {len(df)}")
        print(f"   Bug Rate: {bug_rate:.1f}%")
        print(f"   Avg Velocity: {velocity:.0f} points/sprint")
        print()
    
    print("=" * 60)
    print("RAG Testing Instructions:")
    print("=" * 60)
    print("1. Enable RAG mode in the app")
    print("2. Upload project_alpha_normal.csv → Save as 'Project Alpha'")
    print("3. Upload project_beta_buggy.csv → Save as 'Project Beta'")
    print("4. Upload project_gamma_fast.csv → Save as 'Project Gamma'")
    print()
    print("5. Now upload any project and ask:")
    print("   'How does this compare to past projects?'")
    print("   'What can we learn from similar bug patterns?'")
    print("   'Which past project is most similar to this one?'")
    print()
    print("The AI will reference the specific projects you saved!")
    print("=" * 60)