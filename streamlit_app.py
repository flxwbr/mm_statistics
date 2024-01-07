import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


"""
# Welcome to Streamlit!

Edit `/streamlit_app.py` to customize this app to your heart's desire :heart:.
If you have any questions, checkout our [documentation](https://docs.streamlit.io) and [community
forums](https://discuss.streamlit.io).

In the meantime, below is an example of what you can do with just a few lines of code:
"""

def corr_star_linear(corr_size):
    # attention corr_size must be uneven!
    corr_star = np.ones((corr_size,corr_size))

    return corr_star

def two_point_statistic(np_array, corr_size, iter_number = 1000):
    corr_star = corr_star_linear(corr_size)
    global_results = np.zeros((corr_size, corr_size))

    # get size of image and feasible measurement points
    rows, cols = np_array.shape

    row_points_feas = [corr_size,rows-corr_size]
    col_points_feas = [corr_size,cols-corr_size]
    
    for i in range(iter_number):
        # select random measurment point

        sel_row = np.random.randint(row_points_feas[0],row_points_feas[1]+1)
        sel_col = np.random.randint(col_points_feas[0],col_points_feas[1]+1)    
        position = (sel_row, sel_col)

        # Multiply smaller matrix with the larger matrix at the specified position
        if np_array[sel_row,sel_col] == 1:
            sliced_np_array = np_array[sel_row-corr_size//2:sel_row+corr_size//2+1, sel_col-corr_size//2:sel_col+corr_size//2+1]
            phase_array = sliced_np_array*corr_star

            global_results += phase_array
        else:
            non_phase_array = np.zeros((corr_size,corr_size))

            global_results += non_phase_array

    raw_global_results = global_results.copy()
    global_results = global_results / iter_number

    tps_results = [global_results, corr_size, iter_number, raw_global_results]
    return tps_results

def radial_average(image, center=None):
    if center is None:
        center = np.array(image.shape) // 2

    y, x = np.indices(image.shape)
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    r = r.astype(int)  # Convert to integers for indexing

    radial_sum = np.bincount(r.ravel(), weights=image.ravel())
    radial_count = np.bincount(r.ravel())

    #st.write(r)
    #st.write(r.ravel())
    #st.write(radial_count)
    #st.write(radial_sum)
    radial_avg = radial_sum / radial_count

    #st.write(radial_avg)
    return radial_avg

uploaded_files = st.file_uploader("Choose af file", accept_multiple_files=True)

if uploaded_files is not None:
    file_bytes = np.asarray(bytearray(uploaded_files[0].read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    st.image(opencv_image, caption='Image description')

    gray_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)

    st.image(gray_image, caption='Gray Image')

    Threshold = st.slider('Threshold', 0, 255, 127)

    _, binary_image = cv2.threshold(gray_image, Threshold, 255, cv2.THRESH_BINARY)

    st.image(binary_image, caption='Binary Image')

    my_np_image = np.array(binary_image)

    my_np_image[my_np_image == 0] = 1
    my_np_image[my_np_image == 255] = 0

    max_corr_size = min(my_np_image.shape)

    #st.image(my_np_image, caption='Numpy Image')

    #corr_size = 599

    corr_size = st.slider('Correlation Size', 9, max_corr_size, 599, step=2)
    iter_number = st.slider('Iteration Number', 500, 100000, 1000)

    tps_results = two_point_statistic(my_np_image, corr_size, iter_number)


    custom_cmap = ListedColormap(['#FF0000', '#FFFF00', '#00FF00', '#0000FF'])
    
    fig, ax = plt.subplots()
    heatmap = ax.imshow(tps_results[0])
    cbar = plt.colorbar(heatmap, ticks=[0, 0.05, 0.1, 0.15])
    #cbar.ax.set_yticklabels(['Low', 'Medium', 'High'])
    st.pyplot(fig)
    st.write(np.max(tps_results[0]))

    rad_avg = radial_average(tps_results[0])

    #st.write(rad_avg)
    fig, ax = plt.subplots()
    ax.plot(rad_avg)

    st.pyplot(fig)

    heatmap_df = pd.DataFrame(tps_results[0])
    heatmap_df_csv = heatmap_df.to_csv().encode('utf-8')

    rad_avg_df = pd.DataFrame(rad_avg)
    rad_avg_df_csv = rad_avg_df.to_csv().encode('utf-8')

    st.download_button(label='heatmap raw data', data=heatmap_df_csv, file_name=uploaded_files[0].name.split('.')[0]+'heatmap_raw_pd_df.csv')
    st.download_button(label='radial average raw data', data=rad_avg_df_csv, file_name=uploaded_files[0].name.split('.')[0]+'rad_avg_raw_pd_df.csv')