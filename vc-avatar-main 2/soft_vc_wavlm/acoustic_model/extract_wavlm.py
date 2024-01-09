import os
import glob
import s3prl.hub as hub
import torch
import torchaudio
import numpy as np

from tqdm import tqdm
from sklearn.cluster import KMeans
import pickle
import faiss
#from kmeans_gpu import KMeans

def extract_wavlm(data_path: str, feature_path: str, device: int):
    model_name = "wavlm_large"
    wavlm_model = getattr(hub, model_name)()
    
    device = f"cuda:{device}"
    wavlm_model = wavlm_model.to(device)

    for file in tqdm(glob.glob(os.path.join(data_path, '*.wav'))):
        wav, sr = torchaudio.load(file)
        wav = wav.to(device)

        with torch.no_grad():
            states = wavlm_model(wav)["hidden_states"]
            feature = states[9].squeeze(0)

        feature = feature.cpu().numpy()

        np.save(os.path.join(feature_path, f"{os.path.basename(file)[:-4]}.npy"), feature)

def extract_wavlm_kmeans(wavlm_raw_path: str, wavlm_cluster_path: str,
                         checkpoint_dir: str, train=False):

    wavlm_raw = []
    for file in tqdm(glob.glob(os.path.join(wavlm_raw_path, '*.npy'))):
        wavlm_raw.append(np.load(file))

    ncentroids = 100
    niter = 20
    verbose = True
    d = 1024

    if train:
        wavlm_raw = np.concatenate(wavlm_raw, axis=0)
        print(f"Clustering raw WavLM features with shape: {wavlm_raw.shape}")
        kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu=True)
        kmeans.train(wavlm_raw)

        #kmeans = KMeans(n_clusters=100, verbose=True)

        #kmeans.fit(wavlm_raw)

        #checkpoint_path = os.path.join(checkpoint_dir, f"kmeans_LJ_{100}.pkl")
        #pickle.dump(kmeans, open(checkpoint_path, "wb"))
        checkpoint_path = os.path.join(checkpoint_dir, f"kmeans_LJ_100_faiss.npy")
        np.save(open(checkpoint_path, "wb"), kmeans.centroids)
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"kmeans_LJ_100_faiss.npy")
        #with open(checkpoint_path, "rb") as f:
            #kmeans = pickle.load(f)
        centroids = np.load(checkpoint_path)
        kmeans = faiss.Kmeans(d, ncentroids, niter=0, verbose=verbose, gpu=True)
        kmeans.train(wavlm_raw[0])
        kmeans.centroids = centroids

    for file in tqdm(glob.glob(os.path.join(wavlm_raw_path, '*.npy'))):
        wavlm_raw_file = np.load(file)
        _, wavlm_clustered = kmeans.assign(wavlm_raw_file)
        np.save(os.path.join(wavlm_cluster_path, f"{os.path.basename(file)[:-4]}.npy"), wavlm_clustered)


if __name__ == "__main__":
    extract_wavlm("../../../LJSpeech-1.1/wavs", "../../../LJSpeech-1.1/wavlm_raw/", 0)
    #extract_wavlm_kmeans("../../../LJSpeech-1.1/wavlm_raw", "../../../LJSpeech-1.1/wavlm_clustered/", "kmeans_ckpt", train=False)
    #extract_wavlm_kmeans("../../audios/wavlm_raw/", "../../audios/wavlm_clustered/", "kmeans_ckpt", train=True)