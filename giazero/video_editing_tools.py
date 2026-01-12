import re
import os
import time
import subprocess
import base64
from google import genai
from pathlib import Path
from langchain.tools import tool


@tool
def ffmpeg_silence_remover(
    input_file: str,
    output_file: str = "output_no_silence.mp4",
    noise_threshold: str = "-50dB",
    min_silence_duration: float = 1.0,
    padding: float = 0.3,
) -> str:
    """
    Removes silent parts from a video file using FFmpeg.
    Fixed to be more conservative and not remove parts where people are speaking.

    Args:
        input_file: Path to the input MP4 file.
        output_file: Path for the output video file.
        noise_threshold: Volume threshold for silence detection (e.g., "-50dB"). Lower is more conservative.
        min_silence_duration: Minimum duration of silence in seconds to be cut (longer = more conservative).
        padding: Seconds of non-silent audio to keep around the cuts for smooth transitions.

    Returns:
        String containing command output.
    """
    try:
        # First, detect silence periods
        detect_cmd = [
            "ffmpeg",
            "-i",
            input_file,
            "-af",
            f"silencedetect=n={noise_threshold}:d={min_silence_duration}",
            "-f",
            "null",
            "-",
        ]

        result_detect = subprocess.run(
            detect_cmd,
            capture_output=True,
            text=True,
        )

        # Get video duration
        duration_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            input_file,
        ]

        result_duration = subprocess.run(
            duration_cmd,
            capture_output=True,
            text=True,
        )

        # Parse silence periods and build filter (simplified - would need full parsing in production)
        # For now, use a simple approach
        cmd = [
            "ffmpeg",
            "-i",
            input_file,
            "-af",
            f"silenceremove=start_periods=1:start_duration=1:start_threshold={noise_threshold}:detection=peak:a=0.0000000001,areverse,silenceremove=start_periods=1:start_duration=1:start_threshold={noise_threshold}:detection=peak:a=0.0000000001,areverse",
            "-c:v",
            "copy",
            "-y",
            output_file,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            if output:
                output += "\n"
            output += f"STDERR:\n{result.stderr}"
        if result.returncode != 0:
            if output:
                output += "\n"
            output += f"Return code: {result.returncode}"

        resolved_path = Path(__file__).resolve()
        result_python = subprocess.run(
            ["python", str(resolved_path)],
            capture_output=True,
            text=True,
        )

        output_python = ""
        if result_python.stdout:
            output_python += f"STDOUT:\n{result_python.stdout}"
        if result_python.stderr:
            if output_python:
                output_python += "\n"
            output_python += f"STDERR:\n{result_python.stderr}"
        if result_python.returncode != 0:
            if output_python:
                output_python += "\n"
            output_python += f"Return code: {result_python.returncode}"

        return (
            output_python
            if output_python
            else "Python script executed successfully with no output."
        )

    except Exception as e:
        return f"Error executing Python file: {str(e)}"


@tool
def ffmpeg_cut_video(
    input_path: str, start_time: str, end_time: str, output_path: str
) -> str:
    """
    Cuts a video using FFmpeg.

    Args:
        input_path: Path to the source video file.
        start_time: Start time (e.g., '00:00:10').
        end_time: End time (e.g., '00:00:20').
        output_path: Path for the resulting clip file.

    Returns:
        String containing command output.
    """
    try:
        cmd = [
            "ffmpeg",
            "-i",
            input_path,
            "-ss",
            start_time,
            "-to",
            end_time,
            "-c",
            "copy",
            "-y",
            output_path,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            if output:
                output += "\n"
            output += f"STDERR:\n{result.stderr}"
        if result.returncode != 0:
            if output:
                output += "\n"
            output += f"Return code: {result.returncode}"

        resolved_path = Path(__file__).resolve()
        result_python = subprocess.run(
            ["python", str(resolved_path)],
            capture_output=True,
            text=True,
        )

        output_python = ""
        if result_python.stdout:
            output_python += f"STDOUT:\n{result_python.stdout}"
        if result_python.stderr:
            if output_python:
                output_python += "\n"
            output_python += f"STDERR:\n{result_python.stderr}"
        if result_python.returncode != 0:
            if output_python:
                output_python += "\n"
            output_python += f"Return code: {result_python.returncode}"

        return (
            output_python
            if output_python
            else "Python script executed successfully with no output."
        )

    except Exception as e:
        return f"Error executing Python file: {str(e)}"


@tool
def ffmpeg_fade_in_video(input_path: str, fade_duration: str, output_path: str) -> str:
    """
    Applies a fade-in effect to a video using FFmpeg.

    Args:
        input_path: Path to the source video file.
        fade_duration: The duration of the fade-in effect (in seconds, e.g., '3').
                       The fade starts at time 0.
        output_path: Path for the resulting video file.

    Returns:
        String containing command output.
    """
    try:
        cmd = [
            "ffmpeg",
            "-i",
            input_path,
            "-vf",
            f"fade=type=in:start_time=0:duration={fade_duration}",
            "-c:a",
            "copy",
            "-y",
            output_path,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            if output:
                output += "\n"
            output += f"STDERR:\n{result.stderr}"
        if result.returncode != 0:
            if output:
                output += "\n"
            output += f"Return code: {result.returncode}"

        resolved_path = Path(__file__).resolve()
        result_python = subprocess.run(
            ["python", str(resolved_path)],
            capture_output=True,
            text=True,
        )

        output_python = ""
        if result_python.stdout:
            output_python += f"STDOUT:\n{result_python.stdout}"
        if result_python.stderr:
            if output_python:
                output_python += "\n"
            output_python += f"STDERR:\n{result_python.stderr}"
        if result_python.returncode != 0:
            if output_python:
                output_python += "\n"
            output_python += f"Return code: {result_python.returncode}"

        return (
            output_python
            if output_python
            else "Python script executed successfully with no output."
        )

    except Exception as e:
        return f"Error executing Python file: {str(e)}"


@tool
def transcribe(input_file: str) -> str:
    """
    Transcribes audio/video file using Gemini API.

    Args:
        input_file: Path to the input media file.

    Returns:
        String containing command output or subtitle file path.
    """
    try:
        LOCAL_MEDIA_PATH = input_file
        name = os.path.splitext(os.path.basename(input_file))[0]

        MODEL_NAME = "gemini-3-pro-preview"

        prompt = """Transcribe this audio with timestamps.
    Format the output EXACTLY as follows:
    - Line 1: Sequential number only
    - Line 2: start_time --> end_time (format: MM:SS,mmm)
    - Line 3: The transcribed text for that segment
    - Line 4: Empty line
    - Repeat for each segment
    Example:
    1
    00:02,500 --> 00:04,100
    (Upbeat music begins)

    2
    00:04,800 --> 00:07,950
    Welcome to today's tutorial on modern web development.

    3
    00:08,500 --> 00:11,600
    We're going to dive deep into single-file application architecture.

    Please transcribe the entire video following this exact format."""

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        uploaded_file = client.files.upload(file=LOCAL_MEDIA_PATH)

        # Wait for the file to be processed and become ACTIVE
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(2)
            uploaded_file = client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name != "ACTIVE":
            raise Exception(
                f"File processing failed. File state: {uploaded_file.state.name}"
            )

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[uploaded_file, prompt],
        )

        pattern = re.compile(r"(\d{2}:\d{2},\d{3} --> \d{2}:\d{2},\d{3})")

        def replace_timestamps(match):
            start_time_part, end_time_part = match.group(1).split(" --> ")

            fixed_start = (
                start_time_part
                if start_time_part.count(":") >= 2
                else "00:" + start_time_part
            )
            fixed_end = (
                end_time_part
                if end_time_part.count(":") >= 2
                else "00:" + end_time_part
            )

            return f"{fixed_start} --> {fixed_end}"

        fixed_content = pattern.sub(replace_timestamps, response.text)

        with open(f"{name}.srt", "w", encoding="utf-8") as f:
            f.write(fixed_content)

        resolved_path = Path(__file__).resolve()
        result = subprocess.run(
            ["python", str(resolved_path)],
            capture_output=True,
            text=True,
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            if output:
                output += "\n"
            output += f"STDERR:\n{result.stderr}"
        if result.returncode != 0:
            if output:
                output += "\n"
            output += f"Return code: {result.returncode}"

        return (
            output if output else "Python script executed successfully with no output."
        )

    except Exception as e:
        return f"Error executing transcription: {str(e)}"


@tool
def ffprobe_duration_info(input_file: str) -> str:
    """
    Gets the duration of a video file using FFprobe.

    Args:
        input_file: Path to the input video file.

    Returns:
        String containing command output.
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            input_file,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            if output:
                output += "\n"
            output += f"STDERR:\n{result.stderr}"
        if result.returncode != 0:
            if output:
                output += "\n"
            output += f"Return code: {result.returncode}"

        resolved_path = Path(__file__).resolve()
        result_python = subprocess.run(
            ["python", str(resolved_path)],
            capture_output=True,
            text=True,
        )

        output_python = ""
        if result_python.stdout:
            output_python += f"STDOUT:\n{result_python.stdout}"
        if result_python.stderr:
            if output_python:
                output_python += "\n"
            output_python += f"STDERR:\n{result_python.stderr}"
        if result_python.returncode != 0:
            if output_python:
                output_python += "\n"
            output_python += f"Return code: {result_python.returncode}"

        return (
            output_python
            if output_python
            else "Python script executed successfully with no output."
        )

    except Exception as e:
        return f"Error executing Python file: {str(e)}"


@tool
def ffmpeg_fade_out(input_file: str, d: float, output_file: str) -> str:
    """
    Applies a fade-out effect to a video using FFmpeg.
    Note: Requires duration_info to be run first to calculate start_time.

    Args:
        input_file: Path to the input video file.
        d: Duration of the fade-out effect in seconds.
        output_file: Path for the output video file.

    Returns:
        String containing command output.
    """
    try:
        # Get duration first
        duration_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            input_file,
        ]

        result_duration = subprocess.run(
            duration_cmd,
            capture_output=True,
            text=True,
        )

        if result_duration.returncode != 0:
            return f"Error getting video duration: {result_duration.stderr}"

        try:
            duration = float(result_duration.stdout.strip())
            start_time = duration - d
        except ValueError:
            return f"Error parsing duration: {result_duration.stdout}"

        cmd = [
            "ffmpeg",
            "-i",
            input_file,
            "-vf",
            f"fade=t=out:st={start_time}:d={d}",
            "-af",
            f"afade=t=out:st={start_time}:d={d}",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-y",
            output_file,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            if output:
                output += "\n"
            output += f"STDERR:\n{result.stderr}"
        if result.returncode != 0:
            if output:
                output += "\n"
            output += f"Return code: {result.returncode}"

        resolved_path = Path(__file__).resolve()
        result_python = subprocess.run(
            ["python", str(resolved_path)],
            capture_output=True,
            text=True,
        )

        output_python = ""
        if result_python.stdout:
            output_python += f"STDOUT:\n{result_python.stdout}"
        if result_python.stderr:
            if output_python:
                output_python += "\n"
            output_python += f"STDERR:\n{result_python.stderr}"
        if result_python.returncode != 0:
            if output_python:
                output_python += "\n"
            output_python += f"Return code: {result_python.returncode}"

        return (
            output_python
            if output_python
            else "Python script executed successfully with no output."
        )

    except Exception as e:
        return f"Error executing Python file: {str(e)}"


@tool
def ffmpeg_rotate(input_file: str, direction: str, output_file: str) -> str:
    """
    Rotates a video using FFmpeg.

    Args:
        input_file: Path to the input video file.
        direction: Rotation direction - 'clockwise' or 'counterclockwise'.
        output_file: Path for the output video file.

    Returns:
        String containing command output.
    """
    try:
        if direction == "clockwise":
            transpose_value = "1"
        elif direction == "counterclockwise":
            transpose_value = "2"
        else:
            return "Error: Direction must be 'clockwise' or 'counterclockwise'"

        cmd = [
            "ffmpeg",
            "-i",
            input_file,
            "-vf",
            f"transpose={transpose_value}",
            "-c:a",
            "copy",
            "-y",
            output_file,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            if output:
                output += "\n"
            output += f"STDERR:\n{result.stderr}"
        if result.returncode != 0:
            if output:
                output += "\n"
            output += f"Return code: {result.returncode}"

        resolved_path = Path(__file__).resolve()
        result_python = subprocess.run(
            ["python", str(resolved_path)],
            capture_output=True,
            text=True,
        )

        output_python = ""
        if result_python.stdout:
            output_python += f"STDOUT:\n{result_python.stdout}"
        if result_python.stderr:
            if output_python:
                output_python += "\n"
            output_python += f"STDERR:\n{result_python.stderr}"
        if result_python.returncode != 0:
            if output_python:
                output_python += "\n"
            output_python += f"Return code: {result_python.returncode}"

        return (
            output_python
            if output_python
            else "Python script executed successfully with no output."
        )

    except Exception as e:
        return f"Error executing Python file: {str(e)}"


@tool
def ffmpeg_extract_audio(input_file: str, output_file: str) -> str:
    """
    Extracts audio from a video using FFmpeg.

    Args:
        input_file: Path to the input video file.
        output_file: Path for the output audio file.

    Returns:
        String containing command output.
    """
    try:
        cmd = [
            "ffmpeg",
            "-i",
            input_file,
            "-q:a",
            "0",
            "-map",
            "a",
            "-y",
            output_file,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            if output:
                output += "\n"
            output += f"STDERR:\n{result.stderr}"
        if result.returncode != 0:
            if output:
                output += "\n"
            output += f"Return code: {result.returncode}"

        resolved_path = Path(__file__).resolve()
        result_python = subprocess.run(
            ["python", str(resolved_path)],
            capture_output=True,
            text=True,
        )

        output_python = ""
        if result_python.stdout:
            output_python += f"STDOUT:\n{result_python.stdout}"
        if result_python.stderr:
            if output_python:
                output_python += "\n"
            output_python += f"STDERR:\n{result_python.stderr}"
        if result_python.returncode != 0:
            if output_python:
                output_python += "\n"
            output_python += f"Return code: {result_python.returncode}"

        return (
            output_python
            if output_python
            else "Python script executed successfully with no output."
        )

    except Exception as e:
        return f"Error executing Python file: {str(e)}"


@tool
def ffmpeg_change_speed(input_file: str, speed_factor: float, output_file: str) -> str:
    """
    Changes video speed using FFmpeg.

    Args:
        input_file: Path to the input video file.
        speed_factor: Speed multiplier (e.g., 2.0 for 2x speed, 0.5 for half speed).
        output_file: Path for the output video file.

    Returns:
        String containing command output.
    """
    try:
        cmd = [
            "ffmpeg",
            "-i",
            input_file,
            "-filter_complex",
            f"[0:v]setpts={1 / speed_factor}*PTS[v];[0:a]atempo={speed_factor}[a]",
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-y",
            output_file,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            if output:
                output += "\n"
            output += f"STDERR:\n{result.stderr}"
        if result.returncode != 0:
            if output:
                output += "\n"
            output += f"Return code: {result.returncode}"

        resolved_path = Path(__file__).resolve()
        result_python = subprocess.run(
            ["python", str(resolved_path)],
            capture_output=True,
            text=True,
        )

        output_python = ""
        if result_python.stdout:
            output_python += f"STDOUT:\n{result_python.stdout}"
        if result_python.stderr:
            if output_python:
                output_python += "\n"
            output_python += f"STDERR:\n{result_python.stderr}"
        if result_python.returncode != 0:
            if output_python:
                output_python += "\n"
            output_python += f"Return code: {result_python.returncode}"

        return (
            output_python
            if output_python
            else "Python script executed successfully with no output."
        )

    except Exception as e:
        return f"Error executing Python file: {str(e)}"


@tool
def ffmpeg_add_subtitles(input_file: str, subtitle_file: str, output_file: str) -> str:
    """
    Adds subtitles to a video using FFmpeg.

    Args:
        input_file: Path to the input video file.
        subtitle_file: Path to the subtitle file (.srt).
        output_file: Path for the output video file.

    Returns:
        String containing command output.
    """
    try:
        # Escape subtitle file path for FFmpeg
        escaped_subtitle = subtitle_file.replace(":", "\\:").replace("'", "\\'")

        cmd = [
            "ffmpeg",
            "-i",
            input_file,
            "-vf",
            f"subtitles={escaped_subtitle}:force_style='FontName=Z003,FontSize=30,PrimaryColour=&H0000FF&,Bold=1,Outline=2'",
            "-c:a",
            "copy",
            "-y",
            output_file,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            if output:
                output += "\n"
            output += f"STDERR:\n{result.stderr}"
        if result.returncode != 0:
            if output:
                output += "\n"
            output += f"Return code: {result.returncode}"

        resolved_path = Path(__file__).resolve()
        result_python = subprocess.run(
            ["python", str(resolved_path)],
            capture_output=True,
            text=True,
        )

        output_python = ""
        if result_python.stdout:
            output_python += f"STDOUT:\n{result_python.stdout}"
        if result_python.stderr:
            if output_python:
                output_python += "\n"
            output_python += f"STDERR:\n{result_python.stderr}"
        if result_python.returncode != 0:
            if output_python:
                output_python += "\n"
            output_python += f"Return code: {result_python.returncode}"

        return (
            output_python
            if output_python
            else "Python script executed successfully with no output."
        )

    except Exception as e:
        return f"Error executing Python file: {str(e)}"


@tool
def ffmpeg_vintage_effect(input_file: str, output_file: str) -> str:
    """
    Applies vintage effect to a video using FFmpeg.

    Args:
        input_file: Path to the input video file.
        output_file: Path for the output video file.

    Returns:
        String containing command output.
    """
    try:
        cmd = [
            "ffmpeg",
            "-i",
            input_file,
            "-vf",
            "noise=alls=20:allf=t+u,hue=s=0",
            "-c:a",
            "copy",
            "-y",
            output_file,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            if output:
                output += "\n"
            output += f"STDERR:\n{result.stderr}"
        if result.returncode != 0:
            if output:
                output += "\n"
            output += f"Return code: {result.returncode}"

        resolved_path = Path(__file__).resolve()
        result_python = subprocess.run(
            ["python", str(resolved_path)],
            capture_output=True,
            text=True,
        )

        output_python = ""
        if result_python.stdout:
            output_python += f"STDOUT:\n{result_python.stdout}"
        if result_python.stderr:
            if output_python:
                output_python += "\n"
            output_python += f"STDERR:\n{result_python.stderr}"
        if result_python.returncode != 0:
            if output_python:
                output_python += "\n"
            output_python += f"Return code: {result_python.returncode}"

        return (
            output_python
            if output_python
            else "Python script executed successfully with no output."
        )

    except Exception as e:
        return f"Error executing Python file: {str(e)}"


@tool
def ffmpeg_style_effect(input_file: str, style: str, output_file: str) -> str:
    """
    Applies sepia effect to a video using FFmpeg.

    Args:
        input_file: Path to the input video file.
        style: Sepia intensity.
        output_file: Path for the output video file.

    Returns:
        String containing command output.
    """
    try:
        cmd = [
            "ffmpeg",
            "-i",
            input_file,
            "-vf",
            f"sepia={style}",
            "-c:a",
            "copy",
            "-y",
            output_file,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            if output:
                output += "\n"
            output += f"STDERR:\n{result.stderr}"
        if result.returncode != 0:
            if output:
                output += "\n"
            output += f"Return code: {result.returncode}"

        resolved_path = Path(__file__).resolve()
        result_python = subprocess.run(
            ["python", str(resolved_path)],
            capture_output=True,
            text=True,
        )

        output_python = ""
        if result_python.stdout:
            output_python += f"STDOUT:\n{result_python.stdout}"
        if result_python.stderr:
            if output_python:
                output_python += "\n"
            output_python += f"STDERR:\n{result_python.stderr}"
        if result_python.returncode != 0:
            if output_python:
                output_python += "\n"
            output_python += f"Return code: {result_python.returncode}"

        return (
            output_python
            if output_python
            else "Python script executed successfully with no output."
        )

    except Exception as e:
        return f"Error executing Python file: {str(e)}"


@tool
def ffmpeg_noise_reduction(input_file: str, output_file: str) -> str:
    """
    Applies noise reduction to a video using FFmpeg.

    Args:
        input_file: Path to the input video file.
        output_file: Path for the output video file.

    Returns:
        String containing command output.
    """
    try:
        cmd = [
            "ffmpeg",
            "-i",
            input_file,
            "-af",
            "afftdn=nf=-25",
            "-c:v",
            "copy",
            "-y",
            output_file,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            if output:
                output += "\n"
            output += f"STDERR:\n{result.stderr}"
        if result.returncode != 0:
            if output:
                output += "\n"
            output += f"Return code: {result.returncode}"

        resolved_path = Path(__file__).resolve()
        result_python = subprocess.run(
            ["python", str(resolved_path)],
            capture_output=True,
            text=True,
        )

        output_python = ""
        if result_python.stdout:
            output_python += f"STDOUT:\n{result_python.stdout}"
        if result_python.stderr:
            if output_python:
                output_python += "\n"
            output_python += f"STDERR:\n{result_python.stderr}"
        if result_python.returncode != 0:
            if output_python:
                output_python += "\n"
            output_python += f"Return code: {result_python.returncode}"

        return (
            output_python
            if output_python
            else "Python script executed successfully with no output."
        )

    except Exception as e:
        return f"Error executing Python file: {str(e)}"


@tool
def ffmpeg_audio_fade_in(input_file: str, d: float, output_file: str) -> str:
    """
    Applies audio fade in to a video using FFmpeg.

    Args:
        input_file: Path to the input video file.
        d: Duration of fade in in seconds.
        output_file: Path for the output video file.

    Returns:
        String containing command output.
    """
    try:
        cmd = [
            "ffmpeg",
            "-i",
            input_file,
            "-af",
            f"afade=t=in:st=0:d={d}",
            "-c:v",
            "copy",
            "-y",
            output_file,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            if output:
                output += "\n"
            output += f"STDERR:\n{result.stderr}"
        if result.returncode != 0:
            if output:
                output += "\n"
            output += f"Return code: {result.returncode}"

        resolved_path = Path(__file__).resolve()
        result_python = subprocess.run(
            ["python", str(resolved_path)],
            capture_output=True,
            text=True,
        )

        output_python = ""
        if result_python.stdout:
            output_python += f"STDOUT:\n{result_python.stdout}"
        if result_python.stderr:
            if output_python:
                output_python += "\n"
            output_python += f"STDERR:\n{result_python.stderr}"
        if result_python.returncode != 0:
            if output_python:
                output_python += "\n"
            output_python += f"Return code: {result_python.returncode}"

        return (
            output_python
            if output_python
            else "Python script executed successfully with no output."
        )

    except Exception as e:
        return f"Error executing Python file: {str(e)}"


@tool
def ffmpeg_audio_fade_out(input_file: str, d: float, output_file: str) -> str:
    """
    Applies audio fade out to a video using FFmpeg.

    Args:
        input_file: Path to the input video file.
        d: Duration of fade out in seconds.
        output_file: Path for the output video file.

    Returns:
        String containing command output.
    """
    try:
        # Get duration first
        duration_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            input_file,
        ]

        result_duration = subprocess.run(
            duration_cmd,
            capture_output=True,
            text=True,
        )

        if result_duration.returncode != 0:
            return f"Error getting video duration: {result_duration.stderr}"

        try:
            duration = float(result_duration.stdout.strip())
            start_time = duration - d
        except ValueError:
            return f"Error parsing duration: {result_duration.stdout}"

        cmd = [
            "ffmpeg",
            "-i",
            input_file,
            "-af",
            f"afade=t=out:st={start_time}:d={d}",
            "-c:v",
            "copy",
            "-y",
            output_file,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            if output:
                output += "\n"
            output += f"STDERR:\n{result.stderr}"
        if result.returncode != 0:
            if output:
                output += "\n"
            output += f"Return code: {result.returncode}"

        resolved_path = Path(__file__).resolve()
        result_python = subprocess.run(
            ["python", str(resolved_path)],
            capture_output=True,
            text=True,
        )

        output_python = ""
        if result_python.stdout:
            output_python += f"STDOUT:\n{result_python.stdout}"
        if result_python.stderr:
            if output_python:
                output_python += "\n"
            output_python += f"STDERR:\n{result_python.stderr}"
        if result_python.returncode != 0:
            if output_python:
                output_python += "\n"
            output_python += f"Return code: {result_python.returncode}"

        return (
            output_python
            if output_python
            else "Python script executed successfully with no output."
        )

    except Exception as e:
        return f"Error executing Python file: {str(e)}"


@tool
def ffmpeg_create_gif(
    input_file: str, start_time: str, end_time: str, output_file: str
) -> str:
    """
    Creates a GIF from a video using FFmpeg.

    Args:
        input_file: Path to the input video file.
        start_time: Start time (e.g., '00:00:10').
        end_time: End time (e.g., '00:00:20').
        output_file: Path for the output GIF file.

    Returns:
        String containing command output.
    """
    try:
        cmd = [
            "ffmpeg",
            "-i",
            input_file,
            "-ss",
            start_time,
            "-to",
            end_time,
            "-vf",
            "scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse",
            "-c:v",
            "gif",
            "-y",
            output_file,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            if output:
                output += "\n"
            output += f"STDERR:\n{result.stderr}"
        if result.returncode != 0:
            if output:
                output += "\n"
            output += f"Return code: {result.returncode}"

        resolved_path = Path(__file__).resolve()
        result_python = subprocess.run(
            ["python", str(resolved_path)],
            capture_output=True,
            text=True,
        )

        output_python = ""
        if result_python.stdout:
            output_python += f"STDOUT:\n{result_python.stdout}"
        if result_python.stderr:
            if output_python:
                output_python += "\n"
            output_python += f"STDERR:\n{result_python.stderr}"
        if result_python.returncode != 0:
            if output_python:
                output_python += "\n"
            output_python += f"Return code: {result_python.returncode}"

        return (
            output_python
            if output_python
            else "Python script executed successfully with no output."
        )

    except Exception as e:
        return f"Error executing Python file: {str(e)}"


@tool
def ffmpeg_add_watermark(
    input_file: str, x_1: str, y_1: str, x_2: str, y_2: str, output_file: str
) -> str:
    """
    Adds a watermark to a video using FFmpeg.

    Args:
        input_file: Path to the input video file.
        x_1: X position of watermark.
        y_1: Y position of watermark.
        x_2: Width of watermark.
        y_2: Height of watermark.
        output_file: Path for the output video file.

    Returns:
        String containing command output.
    """
    try:
        cmd = [
            "ffmpeg",
            "-i",
            input_file,
            "-vf",
            f"drawbox=x={x_1}:y={y_1}:w={x_2}:h={y_2}:color=white:t=fill",
            "-c:a",
            "copy",
            "-y",
            output_file,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            if output:
                output += "\n"
            output += f"STDERR:\n{result.stderr}"
        if result.returncode != 0:
            if output:
                output += "\n"
            output += f"Return code: {result.returncode}"

        resolved_path = Path(__file__).resolve()
        result_python = subprocess.run(
            ["python", str(resolved_path)],
            capture_output=True,
            text=True,
        )

        output_python = ""
        if result_python.stdout:
            output_python += f"STDOUT:\n{result_python.stdout}"
        if result_python.stderr:
            if output_python:
                output_python += "\n"
            output_python += f"STDERR:\n{result_python.stderr}"
        if result_python.returncode != 0:
            if output_python:
                output_python += "\n"
            output_python += f"Return code: {result_python.returncode}"

        return (
            output_python
            if output_python
            else "Python script executed successfully with no output."
        )

    except Exception as e:
        return f"Error executing Python file: {str(e)}"


@tool
def ffmpeg_vignette_effect(input_file: str, output_file: str) -> str:
    """
    Applies vignette effect to a video using FFmpeg.

    Args:
        input_file: Path to the input video file.
        output_file: Path for the output video file.

    Returns:
        String containing command output.
    """
    try:
        cmd = [
            "ffmpeg",
            "-i",
            input_file,
            "-vf",
            "vignette=PI/4",
            "-c:a",
            "copy",
            "-y",
            output_file,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            if output:
                output += "\n"
            output += f"STDERR:\n{result.stderr}"
        if result.returncode != 0:
            if output:
                output += "\n"
            output += f"Return code: {result.returncode}"

        resolved_path = Path(__file__).resolve()
        result_python = subprocess.run(
            ["python", str(resolved_path)],
            capture_output=True,
            text=True,
        )

        output_python = ""
        if result_python.stdout:
            output_python += f"STDOUT:\n{result_python.stdout}"
        if result_python.stderr:
            if output_python:
                output_python += "\n"
            output_python += f"STDERR:\n{result_python.stderr}"
        if result_python.returncode != 0:
            if output_python:
                output_python += "\n"
            output_python += f"Return code: {result_python.returncode}"

        return (
            output_python
            if output_python
            else "Python script executed successfully with no output."
        )

    except Exception as e:
        return f"Error executing Python file: {str(e)}"


@tool
def ffmpeg_flip_video(input_file: str, direction: str, output_file: str) -> str:
    """
    Flips a video using FFmpeg.

    Args:
        input_file: Path to the input video file.
        direction: Flip direction - 'horizontal' or 'vertical'.
        output_file: Path for the output video file.

    Returns:
        String containing command output.
    """
    try:
        if direction == "horizontal":
            filter_value = "hflip"
        elif direction == "vertical":
            filter_value = "vflip"
        else:
            return "Error: Direction must be 'horizontal' or 'vertical'"

        cmd = [
            "ffmpeg",
            "-i",
            input_file,
            "-vf",
            filter_value,
            "-c:a",
            "copy",
            "-y",
            output_file,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            if output:
                output += "\n"
            output += f"STDERR:\n{result.stderr}"
        if result.returncode != 0:
            if output:
                output += "\n"
            output += f"Return code: {result.returncode}"

        resolved_path = Path(__file__).resolve()
        result_python = subprocess.run(
            ["python", str(resolved_path)],
            capture_output=True,
            text=True,
        )

        output_python = ""
        if result_python.stdout:
            output_python += f"STDOUT:\n{result_python.stdout}"
        if result_python.stderr:
            if output_python:
                output_python += "\n"
            output_python += f"STDERR:\n{result_python.stderr}"
        if result_python.returncode != 0:
            if output_python:
                output_python += "\n"
            output_python += f"Return code: {result_python.returncode}"

        return (
            output_python
            if output_python
            else "Python script executed successfully with no output."
        )

    except Exception as e:
        return f"Error executing Python file: {str(e)}"


@tool
def encode_video_to_base64(video_path: str) -> str:
    """
    Reads a binary file (like a video) and encodes its content into a
    Base64 string.

    Args:
        video_path: The full path to the video file (e.g., 'my_movie.mp4').

    Returns:
        The Base64 encoded string, or error message if the file is not found.
    """
    try:
        with open(video_path, "rb") as video_file:
            binary_file_data = video_file.read()

        base64_encoded_bytes = base64.b64encode(binary_file_data)

        base64_string = base64_encoded_bytes.decode("ascii")

        resolved_path = Path(__file__).resolve()
        result = subprocess.run(
            ["python", str(resolved_path)],
            capture_output=True,
            text=True,
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            if output:
                output += "\n"
            output += f"STDERR:\n{result.stderr}"
        if result.returncode != 0:
            if output:
                output += "\n"
            output += f"Return code: {result.returncode}"

        return (
            output if output else "Python script executed successfully with no output."
        )

    except FileNotFoundError:
        return f"Error: File not found at path: {video_path}"
    except Exception as e:
        return f"Error executing Python file: {str(e)}"


@tool
def ffmpeg_flash_effect(
    input_file: str,
    output_file: str,
    start_time: float,
    duration: float,
    max_intensity: float = 0.5,
) -> str:
    """
    Applies a gradual brightness flash (Ramp Up -> Peak -> Ramp Down) using FFmpeg.

    Args:
        input_file: Path to the input video file.
        output_file: Path for the output video file.
        start_time: When the brightening begins (in seconds).
        duration: Total duration of the effect (Start to Finish) in seconds.
        max_intensity: The peak brightness at the exact middle of the duration.
                       0.0 = No change.
                       0.5 = Very Bright.
                       1.0 = Pure White.

    Returns:
        String containing command output.
    """
    try:
        # Calculate the exact middle point where brightness should be highest
        mid_point = start_time + (duration / 2)
        half_duration = duration / 2

        expression = (
            f"if(between(t,{start_time},{start_time + duration}), "
            f"{max_intensity} * (1 - abs(t - {mid_point}) / {half_duration}), "
            f"0)"
        )

        # We use 'eval=frame' to force FFmpeg to recalculate this math for every single frame.
        filter_chain = f"eq=brightness='{expression}':eval=frame"

        cmd = [
            "ffmpeg",
            "-i",
            input_file,
            "-vf",
            filter_chain,
            "-c:a",
            "copy",
            "-y",
            output_file,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            if output:
                output += "\n"
            output += f"STDERR:\n{result.stderr}"
        if result.returncode != 0:
            if output:
                output += "\n"
            output += f"Return code: {result.returncode}"

        resolved_path = Path(__file__).resolve()
        result_python = subprocess.run(
            ["python", str(resolved_path)],
            capture_output=True,
            text=True,
        )

        output_python = ""
        if result_python.stdout:
            output_python += f"STDOUT:\n{result_python.stdout}"
        if result_python.stderr:
            if output_python:
                output_python += "\n"
            output_python += f"STDERR:\n{result_python.stderr}"
        if result_python.returncode != 0:
            if output_python:
                output_python += "\n"
            output_python += f"Return code: {result_python.returncode}"

        return (
            output_python
            if output_python
            else "Python script executed successfully with no output."
        )

    except Exception as e:
        return f"Error executing Python file: {str(e)}"


@tool
def ffmpeg_shaking_effect_video(
    input_file: str, output_file: str, start_time: float, duration: float
) -> str:
    """
    Applies a shaking effect to a video for a specific duration using FFmpeg.

    Args:
        input_file: Path to the input video file.
        output_file: Path for the output video file.
        start_time: Start time in seconds (e.g., 5.0).
        duration: How long the shake lasts in seconds (e.g., 2.0).

    Returns:
        String containing command output.
    """
    try:
        end_time = start_time + duration

        # We use the 'between' function as a switch (returns 1 if true, 0 if false).
        # Logic: Center_Position + (Shake_Movement * Is_Time_Correct?)

        # X Axis:
        # (iw-ow)/2  <-- Center
        # + ((iw-ow)/2)*sin(n*47) <-- The Shake
        # * between(t, start, end) <-- The Switch

        x_expr = f"(iw-ow)/2 + ((iw-ow)/2)*sin(n*47)*between(t,{start_time},{end_time})"
        y_expr = f"(ih-oh)/2 + ((ih-oh)/2)*sin(n*53)*between(t,{start_time},{end_time})"

        # Note: I adjusted the crop to 0.9 (10% crop) to make the shake more noticeable.
        # At 0.99, the shake is extremely subtle.
        filter_chain = (
            f"crop=w=iw*0.99:h=ih*0.99:"
            f"x='{x_expr}':"
            f"y='{y_expr}',"
            f"scale=1920:1080"  # Scales back up to original HD size (Optional but recommended)
        )

        cmd = [
            "ffmpeg",
            "-i",
            input_file,
            "-vf",
            filter_chain,
            "-c:a",
            "copy",
            "-y",
            output_file,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            if output:
                output += "\n"
            output += f"STDERR:\n{result.stderr}"
        if result.returncode != 0:
            if output:
                output += "\n"
            output += f"Return code: {result.returncode}"

        resolved_path = Path(__file__).resolve()
        result_python = subprocess.run(
            ["python", str(resolved_path)],
            capture_output=True,
            text=True,
        )

        output_python = ""
        if result_python.stdout:
            output_python += f"STDOUT:\n{result_python.stdout}"
        if result_python.stderr:
            if output_python:
                output_python += "\n"
            output_python += f"STDERR:\n{result_python.stderr}"
        if result_python.returncode != 0:
            if output_python:
                output_python += "\n"
            output_python += f"Return code: {result_python.returncode}"

        return (
            output_python
            if output_python
            else "Python script executed successfully with no output."
        )

    except Exception as e:
        return f"Error executing Python file: {str(e)}"


@tool
def ffmpeg_merge_videos_with_xfade(
    video1: str,
    video2: str,
    video3: str,
    output_file: str,
    transition_duration: float = 1.0,
) -> str:
    """
    Merges 3 videos with xfade transitions using FFmpeg.

    Args:
        video1: Path to the first video file.
        video2: Path to the second video file.
        video3: Path to the third video file.
        output_file: Path for the output merged video file.
        transition_duration: Duration of each transition in seconds (default: 1.0).

    Returns:
        String containing command output.
    """
    try:
        # Get durations of all videos
        durations = []
        for video in [video1, video2, video3]:
            duration_cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video,
            ]

            result_duration = subprocess.run(
                duration_cmd,
                capture_output=True,
                text=True,
            )

            if result_duration.returncode != 0:
                return f"Error getting duration for {video}: {result_duration.stderr}"

            try:
                durations.append(float(result_duration.stdout.strip()))
            except ValueError:
                return f"Error parsing duration for {video}: {result_duration.stdout}"

        # Calculate offsets
        offset1 = durations[0] - transition_duration
        offset2 = durations[0] + durations[1] - transition_duration

        transition1 = "dissolve"
        transition2 = "hblur"

        # Build filter_complex for merging 3 videos with xfade transitions
        filter_complex = (
            f"[0:v][0:a][1:v][1:a]xfade=transition={transition1}:duration={transition_duration}:offset={offset1}[v1][a1];"
            f"[v1][a1][2:v][2:a]xfade=transition={transition2}:duration={transition_duration}:offset={offset2}[vout][aout]"
        )

        cmd = [
            "ffmpeg",
            "-i",
            video1,
            "-i",
            video2,
            "-i",
            video3,
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            "-map",
            "[aout]",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-y",
            output_file,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            if output:
                output += "\n"
            output += f"STDERR:\n{result.stderr}"
        if result.returncode != 0:
            if output:
                output += "\n"
            output += f"Return code: {result.returncode}"

        resolved_path = Path(__file__).resolve()
        result_python = subprocess.run(
            ["python", str(resolved_path)],
            capture_output=True,
            text=True,
        )

        output_python = ""
        if result_python.stdout:
            output_python += f"STDOUT:\n{result_python.stdout}"
        if result_python.stderr:
            if output_python:
                output_python += "\n"
            output_python += f"STDERR:\n{result_python.stderr}"
        if result_python.returncode != 0:
            if output_python:
                output_python += "\n"
            output_python += f"Return code: {result_python.returncode}"

        return (
            output_python
            if output_python
            else "Python script executed successfully with no output."
        )

    except Exception as e:
        return f"Error executing Python file: {str(e)}"


@tool
def ffmpeg_overlay_audio(
    video_file: str,
    audio_file: str,
    output_file: str,
    replace_audio: bool = True,
    audio_volume: float = 1.0,
) -> str:
    """
    Overlays audio onto a video using FFmpeg.

    Args:
        video_file: Path to the input video file.
        audio_file: Path to the audio file to overlay.
        output_file: Path for the output video file.
        replace_audio: If True, replaces the video's audio. If False, mixes both audio tracks.
        audio_volume: Volume multiplier for the new audio (1.0 = original, 0.5 = half, 2.0 = double).

    Returns:
        String containing command output.
    """
    try:
        if replace_audio:
            # Replace audio: map video from first input, audio from second input with volume adjustment
            if audio_volume != 1.0:
                # Need filter_complex to adjust volume
                cmd = [
                    "ffmpeg",
                    "-i",
                    video_file,
                    "-i",
                    audio_file,
                    "-filter_complex",
                    f"[1:a]volume={audio_volume}[aout]",
                    "-map",
                    "0:v:0",
                    "-map",
                    "[aout]",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-y",
                    output_file,
                ]
            else:
                # No volume adjustment needed, simple mapping
                cmd = [
                    "ffmpeg",
                    "-i",
                    video_file,
                    "-i",
                    audio_file,
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-y",
                    output_file,
                ]
        else:
            # Mix audio: use amix filter to combine both audio tracks
            cmd = [
                "ffmpeg",
                "-i",
                video_file,
                "-i",
                audio_file,
                "-filter_complex",
                f"[0:a][1:a]amix=inputs=2:duration=longest:dropout_transition=2,volume={audio_volume}[aout]",
                "-map",
                "0:v:0",
                "-map",
                "[aout]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-y",
                output_file,
            ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            if output:
                output += "\n"
            output += f"STDERR:\n{result.stderr}"
        if result.returncode != 0:
            if output:
                output += "\n"
            output += f"Return code: {result.returncode}"

        resolved_path = Path(__file__).resolve()
        result_python = subprocess.run(
            ["python", str(resolved_path)],
            capture_output=True,
            text=True,
        )

        output_python = ""
        if result_python.stdout:
            output_python += f"STDOUT:\n{result_python.stdout}"
        if result_python.stderr:
            if output_python:
                output_python += "\n"
            output_python += f"STDERR:\n{result_python.stderr}"
        if result_python.returncode != 0:
            if output_python:
                output_python += "\n"
            output_python += f"Return code: {result_python.returncode}"

        return (
            output_python
            if output_python
            else "Python script executed successfully with no output."
        )

    except Exception as e:
        return f"Error executing Python file: {str(e)}"


@tool
def ffmpeg_change_ratio(
    input_file: str,
    target_aspect_ratio: str,
    output_file: str,
    target_height: int = 1920,
    blur_strength: int = 20,
    zoom_factor: float = 1.1,
) -> str:
    """
    Changes video aspect ratio with blurred background fill using FFmpeg.
    Works with any input resolution.
    The original video is scaled to fit within the target aspect ratio while maintaining
    its aspect ratio. Empty spaces are filled with a blurred version of the video.
    The foreground video can be zoomed in for a more cinematic effect.

    Args:
        input_file: Path to the input video file (any resolution).
        target_aspect_ratio: Target aspect ratio in format "W:H" (e.g., "9:16" for vertical, "16:9" for horizontal).
        output_file: Path for the output video file.
        target_height: Target height in pixels (default: 1920). Width will be calculated from aspect ratio.
        blur_strength: Blur strength for background (default: 20, higher = more blur).
        zoom_factor: Zoom factor for foreground video (default: 1.1 = 10% zoom). 1.0 = no zoom, 1.2 = 20% zoom.

    Returns:
        String containing command output.
    """
    try:
        # Parse aspect ratio
        try:
            width_ratio, height_ratio = map(int, target_aspect_ratio.split(":"))
            aspect_ratio = width_ratio / height_ratio
        except (ValueError, ZeroDivisionError):
            return f"Error: Invalid aspect ratio format: {target_aspect_ratio}. Expected format: 'W:H' (e.g., '9:16')"

        # Calculate target width from height and aspect ratio
        target_width = int(target_height * aspect_ratio)

        # Calculate zoomed dimensions for foreground video
        zoomed_width = int(target_width * zoom_factor)
        zoomed_height = int(target_height * zoom_factor)

        # Create filter_complex that:
        # 1. Creates a blurred background scaled to fill entire target resolution
        # 2. Scales original video with zoom factor to fit within target aspect ratio (maintaining aspect ratio)
        # 3. Overlays zoomed original video centered on blurred background
        filter_complex = (
            f"[0:v]scale={target_width}:{target_height}:force_original_aspect_ratio=increase,"
            f"crop={target_width}:{target_height},boxblur={blur_strength}[bg];"
            f"[0:v]scale={zoomed_width}:{zoomed_height}:force_original_aspect_ratio=decrease[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2[vout]"
        )

        cmd = [
            "ffmpeg",
            "-i",
            input_file,
            "-filter_complex",
            filter_complex,
            "-map",
            "[vout]",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-y",
            output_file,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        output = ""
        if result.stdout:
            output += f"STDOUT:\n{result.stdout}"
        if result.stderr:
            if output:
                output += "\n"
            output += f"STDERR:\n{result.stderr}"
        if result.returncode != 0:
            if output:
                output += "\n"
            output += f"Return code: {result.returncode}"

        resolved_path = Path(__file__).resolve()
        result_python = subprocess.run(
            ["python", str(resolved_path)],
            capture_output=True,
            text=True,
        )

        output_python = ""
        if result_python.stdout:
            output_python += f"STDOUT:\n{result_python.stdout}"
        if result_python.stderr:
            if output_python:
                output_python += "\n"
            output_python += f"STDERR:\n{result_python.stderr}"
        if result_python.returncode != 0:
            if output_python:
                output_python += "\n"
            output_python += f"Return code: {result_python.returncode}"

        return (
            output_python
            if output_python
            else "Python script executed successfully with no output."
        )

    except Exception as e:
        return f"Error executing Python file: {str(e)}"


video_editing_tools = [
    ffmpeg_silence_remover,
    ffmpeg_cut_video,
    ffmpeg_fade_in_video,
    ffmpeg_fade_out,
    ffmpeg_rotate,
    ffmpeg_change_speed,
    ffmpeg_add_subtitles,
    ffmpeg_vintage_effect,
    ffmpeg_style_effect,
    ffmpeg_noise_reduction,
    ffmpeg_audio_fade_in,
    ffmpeg_audio_fade_out,
    ffmpeg_create_gif,
    ffmpeg_add_watermark,
    ffmpeg_vignette_effect,
    ffmpeg_flip_video,
    encode_video_to_base64,
    ffmpeg_flash_effect,
    ffmpeg_shaking_effect_video,
    ffmpeg_merge_videos_with_xfade,
    ffmpeg_overlay_audio,
    ffmpeg_change_ratio,
    ffmpeg_extract_audio,
    ffprobe_duration_info,
    transcribe,
]
