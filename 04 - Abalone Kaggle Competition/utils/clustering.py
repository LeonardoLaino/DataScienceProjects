def gmm_tunning(data, n, random_state = 1):
    from sklearn.mixture import GaussianMixture
    import matplotlib.pyplot as plt
    """
    Encontra a melhor configuração para o modelo GMM com base no menor BIC.
    :params:
    data: dados reduzidos após pca, para ajuste do modelo
    n = numero de clusters a serem testados
    """

    n_components = range(1, n+1)
    covariance_type = ['spherical', 'tied', 'diag', 'full']
    best_bic = float('inf')  # Inicialize com um valor infinito
    best_config = None
    bic_scores = []


    for cov in covariance_type:
        hist = []
        for n_comp in n_components:
            gmm = GaussianMixture(n_components=n_comp, covariance_type=cov, random_state= random_state)
            gmm.fit(data)
            bic_score = gmm.bic(data)
            bic_scores.append((cov, n_comp, bic_score))
            hist.append(bic_score)

            if bic_score < best_bic:
                best_bic = bic_score
                best_config = (cov, n_comp)
        
        plt.plot(n_components, hist, marker='o')
        plt.xlabel('Número de Clusters')
        plt.ylabel('BIC')
        plt.title(f'Valor de BIC X Número de Clusters - Covariance Type: {cov}')
        plt.show()


    print("Melhor configuração:", best_config)
    print("Menor BIC:", best_bic)

    return best_config