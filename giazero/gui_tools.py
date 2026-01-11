import base64
import io
import platform
import subprocess
import threading

import pyautogui
from langchain.tools import tool

# Set a pause between PyAutoGUI commands to allow the UI to catch up
pyautogui.PAUSE = 1.0
# Enable failsafe (moving mouse to corner will abort)
pyautogui.FAILSAFE = True

# Global lock to ensure GUI actions are sequential even if called in parallel
gui_lock = threading.Lock()

# Margin in pixels to keep the mouse away from screen edges (avoids Hot Corners)
SAFETY_MARGIN = 20


def _clamp_xy(x: int, y: int) -> tuple[int, int, bool]:
    """Clamp coordinates to be within the safe screen area."""
    width, height = pyautogui.size()

    safe_x = max(SAFETY_MARGIN, min(x, width - SAFETY_MARGIN))
    safe_y = max(SAFETY_MARGIN, min(y, height - SAFETY_MARGIN))

    was_clamped = (safe_x != x) or (safe_y != y)
    return safe_x, safe_y, was_clamped


@tool
def focus_window(app_name: str) -> str:
    """Bring a specific application's window to the front.

    Use this tool if the window loses focus or before typing.

    Args:
        app_name: The name of the application (e.g., "Google Chrome", "Terminal").

    Returns:
        Success message or error.
    """
    try:
        if platform.system() == "Darwin":
            # 'open -a' is a standard macOS command to launch or activate an app
            subprocess.run(["open", "-a", app_name], check=True, capture_output=True)
            return f"Focused application: {app_name}"
        else:
            return "Focusing windows is currently optimized for macOS."
    except subprocess.CalledProcessError as e:
        return f"Error focusing application '{app_name}': {e.stderr if e.stderr else str(e)}"
    except Exception as e:
        return f"Error focusing application: {str(e)}"


@tool
def take_screenshot() -> list:
    """Take a screenshot of the primary screen.

    Returns the screenshot as a base64-encoded image suitable for multimodal models.

    Returns:
        List containing image metadata and base64 data, or an error message.
    """
    try:
        with gui_lock:
            screenshot = pyautogui.screenshot()

        buffer = io.BytesIO()
        screenshot.save(buffer, format="PNG")
        image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return [
            {"type": "text", "text": "Screenshot captured successfully."},
            {
                "type": "image",
                "source_type": "base64",
                "data": image_data,
                "mime_type": "image/png",
            },
        ]
    except Exception as e:
        return [{"type": "text", "text": f"Error taking screenshot: {str(e)}"}]


@tool
def get_screen_info() -> str:
    """Get information about the screen size and current mouse position.

    Returns:
        String with screen resolution and mouse coordinates.
    """
    try:
        with gui_lock:
            width, height = pyautogui.size()
            x, y = pyautogui.position()
        return (
            f"Screen resolution: {width}x{height}\nCurrent mouse position: ({x}, {y})"
        )
    except Exception as e:
        return f"Error getting screen info: {str(e)}"


@tool
def get_mouse_position() -> str:
    """Get the current coordinates of the mouse cursor.

    Returns:
        The (x, y) coordinates.
    """
    try:
        with gui_lock:
            x, y = pyautogui.position()
        return f"Current mouse position: ({x}, {y})"
    except Exception as e:
        return f"Error getting mouse position: {str(e)}"


@tool
def move_mouse(x: int, y: int) -> str:
    """Move the mouse cursor to a specific coordinate.

    Args:
        x: The x-coordinate to move to.
        y: The y-coordinate to move to.

    Returns:
        Success message or error.
    """
    try:
        with gui_lock:
            safe_x, safe_y, clamped = _clamp_xy(x, y)
            pyautogui.moveTo(safe_x, safe_y)

        msg = f"Mouse moved to ({safe_x}, {safe_y})."
        if clamped:
            msg += f" (Clamped from {x}, {y} to avoid screen edges)"
        return msg
    except Exception as e:
        return f"Error moving mouse: {str(e)}"


@tool
def mouse_click(
    x: int = None, y: int = None, button: str = "left", clicks: int = 1
) -> str:
    """Click the mouse.

    Args:
        x: Optional x-coordinate to click at. If None, clicks at current position.
        y: Optional y-coordinate to click at. If None, clicks at current position.
        button: Mouse button to click ('left', 'middle', 'right'). Defaults to 'left'.
        clicks: Number of clicks (e.g., 2 for double-click). Defaults to 1.

    Returns:
        Success message or error.
    """
    try:
        with gui_lock:
            clamped = False
            if x is not None and y is not None:
                safe_x, safe_y, clamped = _clamp_xy(x, y)
                pyautogui.click(x=safe_x, y=safe_y, button=button, clicks=clicks)
                msg = (
                    f"Clicked {button} button {clicks} time(s) at ({safe_x}, {safe_y})."
                )
                if clamped:
                    msg += f" (Clamped from {x}, {y})"
                return msg
            else:
                pyautogui.click(button=button, clicks=clicks)
                curr_x, curr_y = pyautogui.position()
                return f"Clicked {button} button {clicks} time(s) at current position ({curr_x}, {curr_y})."
    except Exception as e:
        return f"Error clicking mouse: {str(e)}"


@tool
def mouse_drag(x: int, y: int, button: str = "left") -> str:
    """Drag the mouse to a specific coordinate while holding a button.

    Args:
        x: The x-coordinate to drag to.
        y: The y-coordinate to drag to.
        button: Mouse button to hold down ('left', 'middle', 'right'). Defaults to 'left'.

    Returns:
        Success message or error.
    """
    try:
        with gui_lock:
            safe_x, safe_y, clamped = _clamp_xy(x, y)
            pyautogui.dragTo(safe_x, safe_y, button=button)

        msg = f"Dragged mouse to ({safe_x}, {safe_y}) with {button} button."
        if clamped:
            msg += f" (Clamped from {x}, {y})"
        return msg
    except Exception as e:
        return f"Error dragging mouse: {str(e)}"


@tool
def scroll(clicks: int) -> str:
    """Scroll the mouse wheel.

    Args:
        clicks: Number of 'clicks' to scroll. Positive for up, negative for down.

    Returns:
        Success message or error.
    """
    try:
        with gui_lock:
            pyautogui.scroll(clicks)
        direction = "up" if clicks > 0 else "down"
        return f"Scrolled {direction} by {abs(clicks)} units."
    except Exception as e:
        return f"Error scrolling: {str(e)}"


@tool
def keyboard_type(text: str, interval: float = 0.1, press_key: str = None) -> str:
    """Type text using the keyboard, optionally pressing a key afterwards.

    Args:
        text: The string to type.
        interval: Seconds to wait between keystrokes. Defaults to 0.1.
        press_key: Optional key to press after typing (e.g., 'enter').

    Returns:
        Success message or error.
    """
    try:
        with gui_lock:
            pyautogui.write(text, interval=interval)
            msg = f"Typed text: '{text}'"
            if press_key:
                pyautogui.press(press_key)
                msg += f" and pressed '{press_key}'"
            return msg
    except Exception as e:
        return f"Error typing text: {str(e)}"


@tool
def keyboard_press(key: str, presses: int = 1) -> str:
    """Press a specific key.

    Args:
        key: The name of the key to press (e.g., 'enter', 'esc', 'tab', 'a', 'b').
             See pyautogui documentation for valid key names.
        presses: Number of times to press the key. Defaults to 1.

    Returns:
        Success message or error.
    """
    try:
        with gui_lock:
            pyautogui.press(key, presses=presses)
        return f"Pressed '{key}' {presses} time(s)."
    except Exception as e:
        return f"Error pressing key: {str(e)}"


@tool
def keyboard_hotkey(keys: str) -> str:
    """Press a combination of keys (hotkey).

    Args:
        keys: Comma-separated string of keys to press together (e.g., 'command,c', 'ctrl,v').

    Returns:
        Success message or error.
    """
    try:
        with gui_lock:
            key_list = [k.strip() for k in keys.split(",")]
            pyautogui.hotkey(*key_list)
        return f"Pressed hotkey: {' + '.join(key_list)}"
    except Exception as e:
        return f"Error pressing hotkey: {str(e)}"


gui_tools = [
    focus_window,
    take_screenshot,
    get_screen_info,
    get_mouse_position,
    move_mouse,
    mouse_click,
    mouse_drag,
    scroll,
    keyboard_type,
    keyboard_press,
    keyboard_hotkey,
]
