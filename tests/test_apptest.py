"""
GridLedger AppTest smoke test.

Tests that the Streamlit app loads all 3 pages without exceptions.
Uses streamlit.testing.v1.AppTest.
"""
from __future__ import annotations

import pytest


def test_app_command_center():
    """App loads Command Center page without exception."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=120)
    # Set default state before running
    at.session_state["active_page"] = "Command Center"
    at.session_state["scenario_preset"] = "Showcase"
    at.session_state["today_day"] = 40
    at.session_state["random_seed"] = 42
    at.session_state["selected_node"] = None
    at.run()
    assert not at.exception, f"Exception on Command Center: {at.exception}"


def test_app_case_file():
    """App loads Case File page without exception."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=120)
    at.session_state["active_page"] = "Case File"
    at.session_state["scenario_preset"] = "Showcase"
    at.session_state["today_day"] = 40
    at.session_state["random_seed"] = 42
    at.session_state["selected_node"] = "T04"
    at.run()
    assert not at.exception, f"Exception on Case File: {at.exception}"


def test_app_scenario_lab():
    """App loads Scenario Lab page without exception."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=120)
    at.session_state["active_page"] = "Scenario Lab"
    at.session_state["scenario_preset"] = "Showcase"
    at.session_state["today_day"] = 40
    at.session_state["random_seed"] = 42
    at.session_state["selected_node"] = None
    at.run()
    assert not at.exception, f"Exception on Scenario Lab: {at.exception}"


def test_app_clean_preset():
    """App loads with Clean preset without exception."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file("app.py", default_timeout=120)
    at.session_state["active_page"] = "Command Center"
    at.session_state["scenario_preset"] = "Clean grid"
    at.session_state["today_day"] = 40
    at.session_state["random_seed"] = 42
    at.session_state["selected_node"] = None
    at.run()
    assert not at.exception, f"Exception on Clean preset: {at.exception}"
