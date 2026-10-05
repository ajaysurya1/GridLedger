from streamlit.testing.v1 import AppTest

at = AppTest.from_file("app.py", default_timeout=120)
at.session_state["active_page"] = "Command Center"
at.session_state["scenario_preset"] = "Showcase"
at.session_state["today_day"] = 40
at.session_state["random_seed"] = 42
at.session_state["selected_node"] = None
at.run()
print("Exception:", at.exception)
if at.exception:
    for exc in at.exception:
        print("EXC:", exc)
