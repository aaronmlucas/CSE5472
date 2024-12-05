import subprocess
import argparse
import os

def capture_traffic(interface, output_file, duration):
    """
    Captures network traffic using Tshark.
    
    Args:
        interface (str): Network interface to capture traffic (e.g., 'eth0', 'wlan0').
        output_file (str): Path to save the .pcap file.
        duration (int): Duration of the capture in seconds.
    """
    try:
        # Check if Tshark is installed
        tshark_path = subprocess.run(["which", "tshark"], capture_output=True, text=True).stdout.strip()
        if not tshark_path:
            raise FileNotFoundError("Tshark is not installed. Please install Tshark and try again.")

        print(f"Starting traffic capture on interface {interface} for {duration} seconds...")
        
        # Run Tshark command to capture traffic
        cmd = [
            "tshark", "-i", interface, 
            "-a", f"duration:{duration}", 
            "-w", output_file
        ]
        subprocess.run(cmd, check=True)
        
        print(f"Traffic capture completed. Data saved to {output_file}.")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture network traffic using Tshark.")
    parser.add_argument("-i", "--interface", required=True, help="Network interface to capture traffic (e.g., eth0, wlan0).")
    parser.add_argument("-o", "--output", required=True, help="Output file to save the .pcap data.")
    parser.add_argument("-d", "--duration", type=int, default=60, help="Duration of traffic capture in seconds (default: 60).")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Start capturing traffic
    capture_traffic(args.interface, args.output, args.duration)
