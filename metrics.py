import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance
from prdc import compute_prdc

def compute_WD(real, generated):
    score = 0
    for i in real.columns:
        # Compute the empirical CDFs
        cdf1,bins1 = np.histogram(real[i].to_numpy(), bins='fd')
        ecdf1 = np.cumsum(cdf1)/len(real)
        ecdf2 = np.cumsum(np.histogram(generated[i].to_numpy(), bins=bins1)[0])/len(generated)
        score += stats.wasserstein_distance(ecdf1, ecdf2)
    return score/len(real.columns)


def compute_JSD(df1, df2):
    # Get set of all variables
    variables = set(df1.columns).union(set(df2.columns))

    # Initialize empty list to store JSD values for each variable
    jsd_values = []

    # Loop through variables
    for var in variables:
        # Get the set of values for the variable from both dataframes
        values1 = set(df1[var].unique())
        values2 = set(df2[var].unique())

        # Create a union of the sets of values for the variable
        all_values = values1.union(values2)

        # Fill missing values with 0
        data1 = df1[var].value_counts().reindex(all_values, fill_value=0)
        data2 = df2[var].value_counts().reindex(all_values, fill_value=0)
        # Compute JSD for the variable and append to list
        jsd = distance.jensenshannon(data1, data2)
        jsd_values.append(jsd)

    return np.mean(jsd_values)

def compute_PCD(real, generated):

    temp = real.to_numpy()
    g = generated.to_numpy()
    scaler = MinMaxScaler().fit(np.concatenate((temp, g)))

    temp = scaler.transform(temp)
    g = scaler.transform(g)
    pcd_r = np.corrcoef(temp.T)
    pcd_g = np.corrcoef(g.T)
    pcd_r[np.isnan(pcd_r)] = 0
    pcd_g[np.isnan(pcd_g)] = 0
    return np.linalg.norm(pcd_r - pcd_g)

def compute_CMD(real, generated):

    all_pairs = [(real.columns[i], real.columns[j]) for i in range(real.shape[1]) for j in range(i + 1, real.shape[1])]
    s=0
    for i in all_pairs:
        contingency_table_r = pd.crosstab(real[i[0]], real[i[1]], dropna=False, normalize=True)
        contingency_table_g = pd.crosstab(generated[i[0]], generated[i[1]], dropna=False, normalize=True)

        # List of all unique values in both variables
        all_categories_0 = sorted(set(real[i[0]].unique()).union(generated[i[0]].unique()))
        all_categories_1 = sorted(set(real[i[1]].unique()).union(generated[i[1]].unique()))

        # Extend the contingencie tables with all the possible values
        contingency_table_r_extended = pd.DataFrame(index=all_categories_0, columns=all_categories_1)
        contingency_table_r_extended.update(contingency_table_r)
        contingency_table_g_extended = pd.DataFrame(index=all_categories_0, columns=all_categories_1)
        contingency_table_g_extended.update(contingency_table_g)
        # Fill missing values with 0
        contingency_table_r_extended = contingency_table_r_extended.fillna(0)
        contingency_table_g_extended = contingency_table_g_extended.fillna(0)

        s+= np.linalg.norm(contingency_table_r_extended - contingency_table_g_extended)
    return s/len(all_pairs)

def transform_discretize(df, i=50):
    for c in df.columns:
        if df[c].nunique()>i and 'IP' not in c and 'Pt' not in c and 'Flags' not in c:
            df[c] = pd.qcut(df[c],i,duplicates='drop')
    return df

def transform_OHE(df, i=50):
    for col in df.columns:
        if df[col].nunique()<i or 'IP' in col or 'Pt' in col or 'Flags' in col:
            df = pd.concat([df,pd.get_dummies(df[col],prefix=col+'_is', prefix_sep='_')],axis=1)
            df=df.drop(col,axis=1)
    return df

def compute_authenticity(train, test, generated, i= 50, n = 500):
    temp = transform_discretize(pd.concat([train, test, generated]), i)

    u= []

    for c in temp.columns:
        u.extend(list(temp[c].unique()))
    u = list(set(u))
    tr = train.replace(dict(zip(u, list(range(len(u))))))
    ts = test.replace(dict(zip(u, list(range(len(u))))))
    g = generated.replace(dict(zip(u, list(range(len(u))))))

    ts = ts.sample(n).to_numpy() ##à tirer ailleurs
    tr = tr.sample(n).to_numpy()
    g = g.sample(n).to_numpy()

    M = np.ones((len(ts)+len(tr), len(g)))
    for i, row in enumerate(np.concatenate([ts, tr])):
        for j, col in enumerate(g):
            M[i, j] = distance.hamming(row, col)
    score = 0
    for r in np.linspace(0,1,15):
        u = M <= r
        result = (np.count_nonzero(u, axis=1) > 0)
        label = np.concatenate([np.zeros(len(ts)), np.ones(len(tr))]).astype(bool)
        if result.sum() == 0:
            continue

        pr = np.logical_and(result, label).sum()/label.sum()
        rr = np.logical_and(result, label).sum()/result.sum()
        f1= 2*pr*rr/(pr+rr)

        score += f1
    return score

def compute_density_coverage(real, g, i=40, n=5):
    temp = transform_OHE(pd.concat([real, g]), i)

    temp = temp.astype(float)

    r = temp.head(len(real)).to_numpy()
    generated = temp.tail(len(g)).to_numpy()

    scaler = MinMaxScaler().fit(np.concatenate((r, generated)))

    r = scaler.transform(r)
    generated = scaler.transform(generated)
    scores = list(compute_prdc(r, generated, n).values())
    return tuple(scores[-2:])

def compute_DKC(u):
    score = 0
    generated = u.astype(str)
    score+=len(generated[((generated["Dst Pt"].isin(['53.0', '137.0', '138.0', '5353.0', '1900.0', '67.0', '0.0', '3544.0', '8612.0', '3702.0', '123.0'])) & (generated["Proto"].str.contains("TCP")))])/len(generated)
    score+=len(generated[((generated["Dst Pt"].isin(['443.0', '80.0', '8000.0', '25.0', '993.0', '587.0', '445.0', '0.0', '84.0', '8088.0', '8080.0'])) & (generated["Proto"].str.contains("UDP")))])/len(generated)
    score+=len(generated[((generated["Dst Pt"] == "0.0") & ((~generated["Proto"].str.contains("ICMP")) | (~generated["Proto"].str.contains("IGMP"))))])/len(generated)
    score+=len(generated[((generated["Proto"].str.contains("ICMP")) & (generated["Out Byte"]!="0.0"))])/len(generated)
    score+=len(generated[((generated["Proto"].str.contains("ICMP")) & (generated["Out Packet"]!="0.0"))])/len(generated)
    score+=len(generated[((generated["Proto"].str.contains("IGMP")) & ((generated["Out Byte"]!="0.0") | (generated["In Byte"]!="0.0")))])/len(generated)
    score+=len(generated[((generated["Dst Pt"].isin(["137.0", "138.0", "1900.0"])) & (generated["In Byte"]!="0.0"))])/len(generated)
    score+=len(generated[((generated["Dst Pt"].isin(["137.0", "138.0", "1900.0"])) & (generated["In Packet"]!="0.0"))])/len(generated)
    score+=len(generated[((generated["Dst Pt"].isin(["137.0", "138.0", "1900.0"])) & (~generated["Dst IP Addr"].str.endswith(".255")))])/len(generated)
    score+=len(generated[(generated["Dst Pt"].isin(["8000.0", "25.0", "443.0","80","587"])) & (generated["Dst IP Addr"].str.startswith("192.168"))])/len(generated)
    score+=len(generated[(generated["Dst Pt"].isin(["993.0", "67.0"])) & (~generated["Dst IP Addr"].str.startswith("192.168"))])/len(generated)
    score+=len(generated[(generated["Dst Pt"]=="53.0") & (generated["Dst IP Addr"] != "DNS")])/len(generated)
    score+=len(generated[(generated["Dst Pt"]=="5353.0") & (generated["Dst IP Addr"] != "10008_251")])/len(generated)
    score+=len(generated[(generated["Flags"]!="......") & (generated["Proto"]!="TCP")])/len(generated)
    score+=len(generated[generated["In Packet"].astype(float)*42 > generated["In Byte"].astype(float)])/len(generated)
    score+=len(generated[generated["Out Packet"].astype(float)*42 > generated["Out Byte"].astype(float)])/len(generated)
    score+=len(generated[generated["In Byte"].astype(float) > 65535*generated["In Packet"].astype(float)])/len(generated)
    score+=len(generated[generated["Out Byte"].astype(float) > 65535*generated["Out Packet"].astype(float)])/len(generated)
    score+=len(generated[generated["Duration"].str.contains("-")])/len(generated)
    score+=len(generated[((generated["Duration"].astype(float)==0) & (generated["In Packet"].astype(float)+generated["Out Packet"].astype(float)>1))])/len(generated)
    score+=len(generated[((generated["Duration"].astype(float)>00) & (generated["In Packet"].astype(float)+generated["Out Packet"].astype(float)==1))])/len(generated)
    return score/20
