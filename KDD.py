import sys
import os
import gzip

cat = ["duration", "protocol_type", "service", "src_bytes", "dst_bytes","flag", "land", "wrong_fragment", "urgent",
        "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations"
        "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
        "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
        "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_same_src_port_rate",
        "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate"]

def parse_log(log_fp):
    log = list()
    eid = 0
    with gzip.open(log_fp, "rt") as ifile:
        for line in ifile:
            event = parse_data(line);
            if event is None:
                continue
            log.append((eid, event))
            eid += 1
    return log

def parse_data(line):
    pairs = line.strip().split(",")
    return dict(zip(cat, pairs))
    

def main():
    if len(sys.argv) != 2:
        print("Please pass in the data set.")
    parse_log(sys.argv[1])

if __name__ == "__main__":
    main()