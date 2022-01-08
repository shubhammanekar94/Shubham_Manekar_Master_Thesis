import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def import_kddcup():

    column_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
        'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
        'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds',
        'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate', 'deu_ruim_ou_nao'
        ]

    raw_df = pd.read_csv('data/raw/corrected.csv', header=None, names=column_names)
    raw_df['deu_ruim_ou_nao']=raw_df['deu_ruim_ou_nao'].str.replace('.','')

    global categorical_features
    categorical_features = list(raw_df.select_dtypes(include=['object']).columns)

    global numerical_features
    numerical_features = list(raw_df.select_dtypes(exclude=['object']).columns)

    return raw_df

def assign_attack_types(raw_df):
    
    DoS = list([
        'smurf', 'pod', 'neptune', 'teardrop', 'land', 'apache2', 
        'back', 'udpstorm', 'mailbomb', 'processtable'
    ])

    U2R = list([

        'buffer_overflow', 'loadmodule', 'perl', 'rootkit', 'xterm', 'ps', 
        'httptunnel', 'sqlattack', 'worm', 'snmpguess'
    ])

    R2L = list([
        'guess_passwd', 'ftp_write', 'imap', 'phf', 'multihop', 'warezmaster',
        'snmpgetattack',  'named', 'xlock', 'xsnoop', 'sendmail'
    ])

    Probe = list([
        'portsweep', 'ipsweep', 'nmap', 'saint', 'satan', 'mscan'
    ])

    for attack in Probe:
        raw_df['deu_ruim_ou_nao']=raw_df['deu_ruim_ou_nao'].str.replace(attack,'Probe')
        
    for attack in DoS:
        raw_df['deu_ruim_ou_nao']=raw_df['deu_ruim_ou_nao'].str.replace(attack,'DoS')

    for attack in U2R:
        raw_df['deu_ruim_ou_nao']=raw_df['deu_ruim_ou_nao'].str.replace(attack,'U2R')

    for attack in R2L:
        raw_df['deu_ruim_ou_nao']=raw_df['deu_ruim_ou_nao'].str.replace(attack,'R2L')

    return raw_df

def label_encoding(raw_df):
    
    lab_enc_df = raw_df.copy()

    labelencoder = LabelEncoder()

    for c in categorical_features:
        lab_enc_df[c] = labelencoder.fit_transform(lab_enc_df[c])

    lab_enc_df.to_csv('data/processed/enc_df.csv',index=False)

    print("Label Encoding Successful! Encoded Data saved as data/processed/enc_df.csv")

    return lab_enc_df

def scaling(lab_enc_df):

    scaled_df = lab_enc_df.copy()

    col_names = numerical_features
    features = scaled_df[col_names]

    scaler = MinMaxScaler().fit(features.values)
    features = scaler.transform(features.values)

    scaled_df[col_names] = features

    scaled_df.to_csv('data/processed/scaled_enc_df.csv',index=False)

    print("Scaling Successful! Scaled Data saved as data/processed/scaled_enc_df.csv")

    return scaled_df