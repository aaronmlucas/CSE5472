import sys
import os
import gzip
import pyshark
from collections import defaultdict

host_stat = defaultdict(lambda: {
    'count': 0,
    'serror_count': 0,
    'rerror_count': 0,
    'same_srv_count': 0
})

def parse_log(log_fp):
    capture = pyshark.FileCapture(log_fp)
    data = []
    for packet in capture:
        data.append(parse_packet(packet))
    calculate_host_stat()
    return data


def parse_packet(packet):
    packet_info = {}
    if 'TCP' in packet:
        packet_info['timestamp'] = packet.sniff_time
        packet_info['protocol'] = packet.transport_layer
        packet_info['tcp_flags'] = packet.tcp.flags
        packet_info['src_port'] = packet.tcp.srcport
        packet_info['dest_port'] = packet.tcp.dstport   
        if packet.tcp.dstport == '80': # Add option for more ports
            host_stat[packet.ip.src]['same_srv_count'] += 1
        if 'SYN' and 'ACK' in packet.tcp.flags: 
            host_stat[packet.ip.src]['serror_count'] += 1
        elif 'RST' in packet.tcp.flags:
            host_stat[packet.ip.src]['rerror_count'] += 1
    if 'IP' in packet:
        packet_info['src'] = packet.ip.src
        host_stat[packet.ip.src]['count'] += 1
        packet_info['dest'] = packet.ip.src
    if 'HTTP' in packet and packet.http.request_method == 'GET':
        packet_info['num_file_creations'] = 1
    return packet_info

def calculate_host_stat():
    for host, stat in host_stat.items():
        stat["serror_count"] = stat["serror_count"] / stat['count'] if stat['count'] > 0 else 0
        stat["rerror_count"] = stat["rerror_count"] / stat['count'] if stat['count'] > 0 else 0
        stat["same_srv_count"] = stat["same_srv_count"] / stat['count'] if stat['count'] > 0 else 0

def main():
    if len(sys.argv) != 2:
        print("Please pass in the data set.")
    print(parse_log(sys.argv[1]))
    print(host_stat)

if __name__ == "__main__":
    main()