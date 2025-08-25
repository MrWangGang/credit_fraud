import subprocess
import sys

if __name__ == "__main__":
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashborad.py", "--server.maxUploadSize=500", "--theme.base", "dark"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"启动 Streamlit 时出错: {e}")