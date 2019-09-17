import numpy as np
import pandas as pd

lower = [-1.55, -115.53, 0.1]
upper = [254.8, 117.47, 1036.9]


# @in N: number of bins 
# @in x_min, x_max: range of the plot
# @in data: list of data arrays
# @in weights: list of weights
# @in if where='post': duplicate the last bin
# @in log=True: return x-axis log
def histHelper(N,x_min,x_max,data,weights=0, where='mid', log=False):
    if log:
        edges = np.logspace(np.log10(x_min), np.log10(x_max), N+1)
    else:
        edges = np.linspace(x_min,x_max,N+1)
    edges_mid = [ edges[i]+(edges[i+1]-edges[i])/2 for i in range(N)]
    bins = [np.histogram(data_i,bins=edges)[0] for data_i in data]
    max_val = [max(x) for x in bins]
    if where=='post':
        bins = [ np.append(b,b[-1]) for b in bins]
    err = np.sqrt(bins)
    if weights!=0:
        bins = [b*s for b,s in zip(bins,weights)]
        err = [e*s for e,s in zip(err,weights)]
        max_val = [v*s for v,s in zip(max_val,weights)]
    return edges, edges_mid, bins, err, max_val
    

# efficiency error unweighted
def effErr(teller,noemer):
    return np.sqrt(teller*(1-teller/noemer))/noemer
 
# weighted histogram error  
def hist_bin_uncertainty(data, weights, bin_edges):
    # Bound the data and weights to be within the bin edges
    in_range_index = [idx for idx in range(len(data)) if data[idx] > min(bin_edges) and data[idx] < max(bin_edges)]
    in_range_data = np.asarray([data[idx] for idx in in_range_index])
    in_range_weights = np.asarray([weights[idx] for idx in in_range_index])

    # Bin the weights with the same binning as the data
    bin_index = np.digitize(in_range_data, bin_edges)
    # N.B.: range(1, bin_edges.size) is used instead of set(bin_index) as if
    # there is a gap in the data such that a bin is skipped no index would appear
    # for it in the set
    binned_weights = np.asarray(
        [in_range_weights[np.where(bin_index == idx)[0]] for idx in range(1, len(bin_edges))])
    bin_uncertainties = np.asarray(
        [np.sqrt(np.sum(np.square(w))) for w in binned_weights])
    return bin_uncertainties

def eventHash(df):
    return df.apply(lambda x: hash(tuple(x)), axis=1)


def inTPC_mask(df, str_x, str_y, str_z, fidvol=[0] * 6):
    global upper,lower
    mask_x = df[str_x].between(lower[0] + fidvol[0], upper[0] - fidvol[1])
    mask_y = df[str_y].between(lower[1] + fidvol[2], upper[1] - fidvol[3])
    mask_z = df[str_z].between(lower[2] + fidvol[4], upper[2] - fidvol[5])
    mask = mask_x & mask_y & mask_z
    return mask


def inTPC_df(df, str_x, str_y, str_z, fidvol=[0] * 6):
    mask = inTPC_mask(df, str_x, str_y, str_z, fidvol)
    return df[mask]


def cosmic_angles(mom_x, mom_y, mom_z, df_out=True):
    theta_beam = np.arctan2(np.sqrt(np.square(mom_x) + np.square(mom_y)), mom_z)
    phi_beam = np.arctan2(mom_y, mom_x)
    if df_out:
        return pd.DataFrame({"theta": theta_beam, "phi": phi_beam})[["theta", "phi"]]
    else:
        return np.array([theta_beam, phi_beam])


def mc_track_contained(x_s, y_s, z_s, p_x, p_y, p_z, length):
    l_n = length / np.sqrt(p_x ** 2 + p_y ** 2 + p_z ** 2)
    x_e = x_s + p_x * l_n
    y_e = y_s + p_y * l_n
    z_e = z_s + p_z * l_n
    df_end = pd.DataFrame({"x_e": x_e, "y_e": y_e, "z_e": z_e})
    return inTPC_mask(df_end, "x_e", "y_e", "z_e")
