import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

st.set_page_config(page_title="KMeans Clustering App", page_icon=":bar_chart:", layout="wide")

st.title("🔍 KMeans Clustering App")

# Sidebar: File upload
st.sidebar.header("Upload your CSV file")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

# Sidebar: Number of clusters
k = st.sidebar.slider("Select number of clusters", 2, 10, 3)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Display the dataframe
    st.write("### Dataset Preview:")
    st.write(df.head())

    # Sidebar: Column selection
    st.sidebar.header("Select columns for clustering")
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    selected_columns = st.sidebar.multiselect("Select the numeric columns to use for clustering", numeric_columns)

    if len(selected_columns) > 1:
        # KMeans clustering
        X = df[selected_columns].values
        kmeans = KMeans(n_clusters=k, init="k-means++")
        clusters = kmeans.fit_predict(X)
        df['Cluster'] = clusters

        # Display cluster centers
        st.write("### Cluster Centers:")
        st.write(pd.DataFrame(kmeans.cluster_centers_, columns=selected_columns))

        # Plotting 2D scatter plot with cluster centers
        st.write("### 2D Scatter Plot with Cluster Centers:")
        fig, ax = plt.subplots()
        scatter = ax.scatter(X[:, 0], X[:, 1], c=clusters, cmap='rainbow', s=100, alpha=0.7, edgecolors='k')
        ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black', marker='x', label='Centroids')
        plt.xlabel(selected_columns[0])
        plt.ylabel(selected_columns[1])
        plt.title("2D Clustering with Centroids")
        plt.legend()
        st.pyplot(fig)

        # Plotting 3D scatter plot with cluster centers (if more than 2 columns are selected)
        if len(selected_columns) == 3:
            st.write("### 3D Scatter Plot with Cluster Centers:")
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(df[selected_columns[0]], df[selected_columns[1]], df[selected_columns[2]], c=clusters, cmap='rainbow', s=100, alpha=0.7, edgecolors='k')
            ax.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s=300, c='black', marker='x', label='Centroids')
            ax.set_xlabel(selected_columns[0])
            ax.set_ylabel(selected_columns[1])
            ax.set_zlabel(selected_columns[2])
            plt.title("3D Clustering with Centroids")
            plt.legend()
            st.pyplot(fig)

        # Enhanced Visualization: Pairplot
        st.write("### Pairplot of Clusters:")
        pairplot_fig = sns.pairplot(df, hue='Cluster', palette='rainbow', vars=selected_columns)
        st.pyplot(pairplot_fig)
        
        # Enhanced Visualization: Cluster count
        st.write("### Cluster Distribution:")
        cluster_counts = df['Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        st.bar_chart(cluster_counts.set_index('Cluster'))

    else:
        st.warning("Please select at least 2 numeric columns for clustering.")
else:
    st.info("Please upload a CSV file to begin.")
