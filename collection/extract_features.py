import pyshark
import csv
import argparse
from collections import defaultdict

def classify_flag(connection_packets):
    """
    Classify the connection's flag based on the sequence of TCP packets.
    """
    has_syn = has_syn_ack = has_fin = has_rst = False
    originator_to_responder_data = False
    responder_to_originator_data = False

    for packet in connection_packets:
        if 'tcp' in packet:
            tcp_flags = int(packet.tcp.flags, 16)
            if tcp_flags & 0x02:  # SYN
                has_syn = True
            if tcp_flags & 0x12:  # SYN-ACK
                has_syn_ack = True
            if tcp_flags & 0x01:  # FIN
                has_fin = True
            if tcp_flags & 0x04:  # RST
                has_rst = True
            if int(packet.length) > 0:
                if packet.ip.src == connection_packets[0].ip.src:
                    originator_to_responder_data = True
                else:
                    responder_to_originator_data = True

    # Classify based on observed events
    if has_syn and not has_syn_ack:
        return "S0"
    elif has_syn and has_syn_ack and not (originator_to_responder_data or responder_to_originator_data):
        return "S1"
    elif has_syn and has_syn_ack and originator_to_responder_data and not responder_to_originator_data:
        return "S2"
    elif has_syn and has_syn_ack and originator_to_responder_data and responder_to_originator_data:
        if has_fin:
            return "SF"
        else:
            return "S3"
    elif has_rst and not (originator_to_responder_data or responder_to_originator_data):
        return "RSTOS0"
    elif has_rst and originator_to_responder_data:
        return "RSTO"
    elif has_rst and responder_to_originator_data:
        return "RSTR"
    elif has_syn and has_fin and not originator_to_responder_data:
        return "SH"
    elif has_rst:
        return "REJ"
    else:
        return "OTH"

def extract_duration(connection_packets):
    """
    Calculate the total duration of the connection.
    """
    try:
        start_time = float(connection_packets[0].sniff_timestamp)
        end_time = float(connection_packets[-1].sniff_timestamp)
        return max(0, round(end_time - start_time))  # Ensure non-negative
    except Exception:
        return 0

def extract_src_bytes(connection_packets):
    """
    Calculate the total bytes sent from source to destination.
    """
    try:
        return sum(int(packet.length) for packet in connection_packets if packet.ip.src == connection_packets[0].ip.src)
    except Exception:
        return 0

def extract_dst_bytes(connection_packets):
    """
    Calculate the total bytes sent from destination to source.
    """
    try:
        return sum(int(packet.length) for packet in connection_packets if packet.ip.dst == connection_packets[0].ip.src)
    except Exception:
        return 0

def extract_features(pcap_file, output_csv):
    """
    Extract features from a .pcap file and save them to a CSV file in KDDCup-compatible order.
    """
    print(f"Processing {pcap_file}...")
    features = []

    # Group packets by connection (5-tuple)
    connections = defaultdict(list)
    try:
        cap = pyshark.FileCapture(pcap_file)

        # Group packets into connections
        for packet in cap:
            if 'ip' in packet and 'tcp' in packet:
                key = (
                    packet.ip.src,
                    packet.ip.dst,
                    packet.tcp.srcport,
                    packet.tcp.dstport,
                    packet.transport_layer
                )
                connections[key].append(packet)

        # Process each connection
        for key, connection_packets in connections.items():
            feature = {
                'duration': extract_duration(connection_packets),
                'protocol_type': key[4],
                'service': 'unknown',  # Placeholder until service mapping is added
                'src_bytes': extract_src_bytes(connection_packets),
                'dst_bytes': extract_dst_bytes(connection_packets),
                'flag': classify_flag(connection_packets),
                'land': 0,  # Placeholder; set to 1 if src_ip == dst_ip
            }
            features.append(feature)

        cap.close()

        # Save features to CSV
        fieldnames = ['duration', 'protocol_type', 'service', 'src_bytes', 'dst_bytes', 'flag', 'land']
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(features)

        print(f"Feature extraction completed. Saved to {output_csv}.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract KDDCup-compatible features from a .pcap file.")
    parser.add_argument("-i", "--input", required=True, help="Path to the input .pcap file.")
    parser.add_argument("-o", "--output", required=True, help="Path to save the extracted features as a CSV file.")
    args = parser.parse_args()

    extract_features(args.input, args.output)
