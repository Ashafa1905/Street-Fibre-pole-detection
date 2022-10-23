import streamlit as st
import pandas as pd
import cv2
import PIL
from PIL import Image, ImageEnhance, ImageTk
import numpy as np
import os
import requests
import random
import onnxruntime as ort
from pathlib import Path
from collections import OrderedDict,namedtuple
from streamlit_option_menu import option_menu
import time, sys
from streamlit_embedcode import github_gist
import urllib.request
import urllib
import matplotlib.pyplot as plt
import geopy
from geopy.geocoders import Nominatim
import glob
import osmnx as ox
from shapely import wkt
from IPython.display import Image, display
import tqdm
from tqdm._tqdm_notebook import tqdm_notebook
from geopy.extra.rate_limiter import RateLimiter
import plotly_express as px
from PIL import Image, ImageEnhance, ImageTk
from PIL import Image, ImageEnhance, ImageTk




@st.cache
def load_image(image_file):
        img = Image.open(image_file)
        return img


def save_uploadedfile(uploadedfile):
     with open(os.path.join("tempDir", uploadedfile.name),"wb") as f:
         f.write(uploadedfile.getbuffer())
     return st.success("Saved File:{} to tempDir".format(uploadedfile.name))
path = r"C:\Users\CHIBUIKEM EFUGHA\OneDrive\Documents\streamlit object detection\Content\\"
path1 = r"C:\Users\CHIBUIKEM EFUGHA\OneDrive\Documents\streamlit object detection\tempDir2\\"
def file_delete(path):
    for file_name in os.listdir(path):
        # construct full file path
        file = path + file_name
        if os.path.isfile(file):
#             print('Deleting file:', file)
            os.remove(file)




def Detection(our_image):
    # Name of the classes according to class indices.
    # Make sure all your classes are in this list in the right order!!
    
    #img = cv2.imread("C:/Users/CHIBUIKEM EFUGHA/OneDrive/Documents/streamlit object detection/tempDir/image_file")
    names = ["fibre-pole"]

    # Creating random colors for bounding box visualization.
    colors = {
        name: [random.randint(0, 255) for _ in range(3)] for i, name in enumerate(names)
    }
    img = cv2.imread(our_image)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocessing the image for prediction.
    image = img.copy()
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)

    im = image.astype(np.float32)
    im /= 255
    # im.shape

    # Getting onnx graph input and output names.
    outname = [i.name for i in session.get_outputs()]
    inname = [i.name for i in session.get_inputs()]
    inp = {inname[0]: im}

    # Running inference using session.
    outputs = session.run(outname, inp)[0]


    ori_images = [img.copy()]

    # Visualizing bounding box prediction.
    for i, (batch_id, x0, y0, x1, y1, cls_id, score) in enumerate(outputs):
        image = ori_images[int(batch_id)]
        box = np.array([x0, y0, x1, y1])
        box -= np.array(dwdh * 2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score), 3)
        name = names[cls_id]
        color = colors[name]
        name += " " + str(score)
        cv2.rectangle(image, box[:2], box[2:], color, 2)
        cv2.putText(
            image,
            name,
            (box[0], box[1] - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            [225, 255, 255],
            thickness=2,
        )
    return Image.fromarray(ori_images[0])




#sample point function
def sample_points(G, n):
    gdf_edges = ox.graph_to_gdfs(G, nodes=False)[['geometry', 'length']]
    weights = gdf_edges['length'] / gdf_edges['length'].sum()
    idx = np.random.choice(gdf_edges.index, size=n, p=weights)
    lines = gdf_edges.loc[idx, 'geometry']
    return lines.interpolate(lines.length * np.random.rand())


#function to get the location coordinates
def get_coords(address, buffer=500):

  
  gdf = ox.geocode_to_gdf(address, buffer_dist=buffer, which_result=1)
  polygon=gdf.geometry[0]
  if polygon.type=='Polygon':

    G = ox.project_graph(ox.graph_from_polygon(polygon, network_type='drive'))
    points = sample_points(G, 100)
    fig, ax = ox.plot_graph(G, show=False, close=False )
    points.plot(ax=ax)
    
    #plt.savefig('/content/polygon.png')
    coords=[]
    nodes = list(G.nodes)
    for node in nodes:
      lat=G.nodes[node]['lat']
      lon=G.nodes[node]['lon']
      coords.append(f'{lat},{lon}')

  return coords


#function to download street view images
def ups_download(address, api_key, buffer = 500):
    """accept a list of addresses, download google street view images, save the images, and return image paths

        args:
            locations: list of cordinates or location addresses
            api_key: your google static street view api key

    """
  
    coords = get_coords(address, buffer=500)
    base_url = 'https://maps.googleapis.com/maps/api/streetview?'
    i=0
    for location in coords:
      pic_params = {'key': api_key, 'location': location, 'heading': 90, 'pitch': -0.76, 'fov': 120,
                      'size': "640x640"}
      name=location.replace(',','_')  
      pic_response = requests.get(base_url, params=pic_params)
      with open(f'/Users/CHIBUIKEM EFUGHA/OneDrive/Documents/streamlit object detection/Content/{name}.jpg', 'wb') as file:
          file.write(pic_response.content)
          pic_response.close()
    return address



#function that helps to download a street view images in different views.
def ups_download_singleAddress(address, api_key):
    
    """acceept list of addresses, download google street view images, save the images, and return image paths
        
        args:
            locations: list of cordinates or location addresses
            api_key: your google static street view api key
    
    """
    locator = Nominatim(user_agent="myGeocoder")
    locate = locator.geocode(address)
    #print("Latitude = {}, Longitude = {}".format(location.latitude, location.longitude))
    lat= locate.latitude
    lon= locate.longitude
    latlon = f"{lat},{lon}"
    print(latlon)
    
    base_url = 'https://maps.googleapis.com/maps/api/streetview?'
    #i=0
    angles = [0,90,180,270]
    for angle in range(len(angles)):
      pic_params = {'key': api_key,'location': latlon,'heading': angles[angle],'pitch': -0.76,'fov': 120,'size': "640x640"}
      name=latlon.replace(",",str(angles[angle]))
      print(name)
      pic_response = requests.get(base_url, params=pic_params)
      with open(f'/Users/CHIBUIKEM EFUGHA/OneDrive/Documents/streamlit object detection/tempDir2/{name}.jpg', 'wb') as file:
          file.write(pic_response.content)
          pic_response.close()
          #i+=1
    return address





cuda = False
w = "C:/Users/CHIBUIKEM EFUGHA/OneDrive/Documents/streamlit object detection/pole_model.onnx"
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
session = ort.InferenceSession(w, providers=providers)


def letterbox(
    im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32
):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )  # add border
    return im, r, (dw, dh)
        

downloaded_image_path1 = '/Users/CHIBUIKEM EFUGHA/OneDrive/Documents/streamlit object detection/tempDir2/*.jpg'
def Detection_downloaded_single(image_path):
    total_images=0
    images_with_poles=0
    for file in glob.glob(image_path):
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        image = img.copy()
        image, ratio, dwdh = letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im /= 255
        # im.shape

        outname = [i.name for i in session.get_outputs()]
        inname = [i.name for i in session.get_inputs()]
        inp = {inname[0]:im}
        outputs = session.run(outname, inp)[0]

        total_images+=1
        if len(outputs)==0:
            data_dic['Fibre_Pole_Present'].append('No')
            images_with_poles=images_with_poles
        else:
            data_dic['Fibre_Pole_Present'].append('Yes')
            images_with_poles+=1
    return images_with_poles, total_images



data_dic={'Latitude':[], 'Longitude':[], 'Address':[], 'Fibre_Pole_Present':[]}
downloaded_image_path = '/Users/CHIBUIKEM EFUGHA/OneDrive/Documents/streamlit object detection/content/*.jpg'
def Detection_downloaded(image_path, region_name):
    total_images=0
    images_with_poles=0
    for file in glob.glob(image_path):
        img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
        image = img.copy()
        image, ratio, dwdh = letterbox(image, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)

        im = image.astype(np.float32)
        im /= 255
        # im.shape

        outname = [i.name for i in session.get_outputs()]
        inname = [i.name for i in session.get_inputs()]
        inp = {inname[0]:im}
        outputs = session.run(outname, inp)[0]

         #retrieve latitude and longitude
        
        lat = file[:-4].split('/')[-1].split('\\')[-1].split('_')[0]
        lon=file[:-4].split('/')[-1].split('_')[1]
        
        data_dic['Latitude'].append(lat)
        data_dic['Longitude'].append(lon)


        # convert coordinate to an address
        locator = Nominatim(user_agent="myGeocoder")
        coordinates = f'{lat},{lon}'
        location = locator.reverse(coordinates)
        data_dic['Address'].append(location.address)

        total_images+=1
        if len(outputs)==0:
            data_dic['Fibre_Pole_Present'].append('No')
            images_with_poles=images_with_poles
        else:
            data_dic['Fibre_Pole_Present'].append('Yes')
            images_with_poles+=1

    df = pd.DataFrame(data_dic)
    df.to_csv(f'C:/Users/CHIBUIKEM EFUGHA/OneDrive/Documents/streamlit object detection/data_csv/{region_name}.csv', index=False, header=True)
    df_load = pd.read_csv(f'C:/Users/CHIBUIKEM EFUGHA/OneDrive/Documents/streamlit object detection/data_csv/{region_name}.csv')
    return images_with_poles, total_images, df, df_load



def main():
    """Fibre Utility Pole Detection App """
    # st.set_page_config(page_title="Fibre Pole Detection", page_icon=":chart_with_upwards_trend:", )

    # Creates a main title and subheader on your page -
    # logo = Image.open("resources/imgs/tweet_logo.png")
    # st.image(logo)
    #st.title("Eco")
    # st.subheader("Climate change tweet classification")

    # Design horizontal bar
    menu = ["Home", "Task", "About Us"]
    selection = option_menu( menu_title=None,
    options=menu,
    icons=["house", "list-task", "textarea-t",  "file-person"],
    orientation='horizontal',
    styles={
                "container": {"padding": "0!important"},
                "icon": {"color": "orange", "font-size": "25px",  },
                "nav-link": {
                    "font-size": "20px",
                    "text-align": "center",
                    "margin": "5px",
                    "--hover-color": "#eee",
                },
                "nav-link-selected": {"background-color": "green"},
            },
        ) 
    
    if selection == "Home":
        st.title("Fibre Utility Pole Detection")
        st.image("https://media.discordapp.net/attachments/1009195330354548767/1033532572447277157/backagroud.jpg")
        #st.image("https://img.freepik.com/premium-photo/technician-internet-service-provider-is-checking-fiber-optic-cables-after-install-electric-pole_40313-616.jpg?w=2000")
        new_title = '<p style="font-size: 42px;">Welcome to our Object Detection App!</p>'
        read_me_0 = st.markdown(new_title, unsafe_allow_html=True)

        read_me = st.markdown("""
        This project was built using Streamlit and OpenCV 
        to demonstrate YOLOV7 Object detection in both Street addresses
        and images.
        
         A total of about 500 street view images of fibre utility poles were used to train and validate a YOLOV7 pytorch model. Images were obtained from roboflow in COCO format. 
        These images were annotated and labeled using a rectangular shape method on roboflow before being used for training. Model was further pretrained with more images gotten from street shots.   
        After training model was exported to ONNX for esasy integration into the streamlit platform and for faster computing.

        Model Benefits: 1. This model is associated with a high perorming accuracy which can be able to detect distant fibre pole objects.
        2. Model is very fast, can run inference on over 70 images within 2-3mins.
        3. Model has a high predicting accuracy.
        """)

        new_title_1 = '<p style="font-size: 42px;">How this App Works</p>'
        read_me_1 = st.markdown(new_title_1, unsafe_allow_html=True)
        read_me_2 = st.markdown(""" This App is an object detection app used to detect Fibre Utility Poles from any google street address provided. It extensively downloads street street 
        images along the region/province provided from different country and detects the presence of fibre utility poles from the address provided. In summary, it the app detects poles and provides the summary statistics 
        of the number of poles present in the address provided. 
        App Benefits: 
        a. This particular app is user friendly.
        b. It is a fast operating app which can easily be integrated.
        c. For each task chosen, app archives the previous selection and hence makes the present selection faster.""")
    
        new_title2 = '<p style="font-size: 22px;">The YOLO Model Architecture</p>'
        read_me_3 = st.markdown(new_title2, unsafe_allow_html=True)

        st.image("https://blog.roboflow.com/content/images/2022/07/image-33.webp")

        read_me_4 = st.markdown("""If you want to know more about yolov7 model, you can watch the video below""")
        
        st.video("https://www.youtube.com/watch?v=OFggvqHy_5Y")

    if selection == "Task":
        activities = ["Utility Pole Detection(Image)", "Utility Pole Detection(Street Address)", "Support System"]
        choice =  st.sidebar.selectbox("Select Activity", activities)
        if choice == "Utility Pole Detection(Image)":
            st.title("Fibre Pole Detection by image upload")
            st.image("https://media.socastsrm.com/wordpress/wp-content/blogs.dir/2798/files/2021/12/ubiquity-to-build-open-access-fiber-optic-infrastructure-across-georgetown-texas.png")

            st.markdown("""Instructions: Upload an image in the uploading section, click on the process button and wait for the app to carry out the task.
            If you ain't satisfied with this solution, please go to the select activity for other functionality settings.
            """)

            st.subheader("Image Upload")

            image_file = st.file_uploader("Upload Image", type = ["jpg", "png", "jpeg"])

            if image_file is not None:
                file_details = {"filename":image_file.name, "filetype":image_file.type,
                                "filesize":image_file.size}
                st.text("Original Image")
                st.image(load_image(image_file), width=300)
                #Saving upload
                with open(os.path.join("tempDir", image_file.name),"wb") as f:
                    f.write((image_file).getbuffer())
                st.success("Upload Successful")
            
        
            if st.button("Process"):
                img_placeholder = st.empty()
                    
                try:
                    image_path = 'tempDir\{}'.format(file_details['filename'])
                    st.text('Detecting fibre Pole, please wait.....:)')
                    predicted_image = Detection(image_path)
                    # Display the image with the detections in the Streamlit app
                    img_placeholder.image(predicted_image, channels='BGR', width=300)
                    st.success('Process Completed')
                    st.write("You can try out other service options")
                    

                except KeyboardInterrupt:
                    pass
                

        if choice ==   "Utility Pole Detection(Street Address)":
            st.title("Detection of Poles in Google Streets")
            st.image("https://www.howtogeek.com/wp-content/uploads/2021/04/google-maps-split-screen.png?width=1198&trim=1,1&bg-color=000&pad=1,1")
            st.subheader("Street View Download")
            st.markdown("""*Instructions: Choose the method of image download from street, Single street download or select the region 
            from the provided regional addresses or manually type in the area of your choice.Then, Click the download button and wait for the app to carry out the function. Once successfully downloaded, then click 
            on the predict button for the app to detect the number of poles present in the images downloaded from the chosen address. This instruction applies to any of the options provided for the address input.*""")

            input_type = st.radio("Choose Address options or type in your address", ('Input Single Street Address', 'Choose from Provided Region', 'Input Region'))
        
            if input_type == 'Input Single Street Address':
                Address = st.text_input('Type in address: ','E.g; Street name, Region/Province, country')
                api_key = st.text_input("Enter your API key: eg, AIzaSyCpF1uPQLSCrx2CdsLLF_db")
                
                if st.button("Download images"):
                    file_delete(path1)

                    try: 
                        ups_download_singleAddress(Address, api_key)
                        _, _, files = next(os.walk("/Users/CHIBUIKEM EFUGHA/OneDrive/Documents/streamlit object detection/tempDir2"))
                        file_count = len(files)
                        st.success(f'{file_count} images downloaded successfully from the address provided')


                    except KeyboardInterrupt:
                        pass

                if st.button("Predict"):
                    predicted_images_with_pole, total_images_folder = Detection_downloaded_single(downloaded_image_path1)
                    st.success(f'There are {predicted_images_with_pole} images with poles from {total_images_folder} different angles of street view')




            elif input_type == 'Choose from Provided Region':
                Address = st.selectbox(
                'Choose a Region:',
                ('Select a region', '2225 Market St, Johannesburg, Gauteng, South Africa', '2027 Old Cres, Lady Frere, Eastern Cape, South Africa',
                '1325 Burger St, Sibasa, Limpopo, South Africa','954 Union Lane, Hillcrest, KwaZulu-Natal, South Africa',
                '123 Prospect St, Pretoria, Gauteng, South Africa','1321 Rissik St, Witbank, Mpumalanga, South Africa',
                '1769 Broad Rd, Vanderbijlpark, Gauteng, South Africa','1082 Bezuidenhout St, Bethal, Mpumalanga, South Africa',
                '1441 Fox St, Stilfontein, North West, South Africa','1472 Bodenstein St, Boksburg, Gauteng, South Africa',
                '1111 Bodenstein St, Boksburg, Gauteng, South Africa','589 Plein St, Ottery, Western Cape, South Africa',
                '245 Nelson Mandela Drive, Pietersburg, Limpopo, South Africa','1230 Old Cres, Whittlesea, Eastern Cape, South Africa',
                '706 South St, Mabopane, Gauteng, South Africa','194 Bath Rd, Ndwedwe, KwaZulu-Natal, South Africa',
                '1299 Gray Pl, Umlazi, KwaZulu-Natal, South Africa','1730 Thomas St, Pietermaritzburg, KwaZulu-Natal, South Africa',
                '1542 Plein St, Mitchells Plain, Western Cape, South Africa','298 Thutlwa St, Letaba, Limpopo, South Africa',
                '10, Ikomi St, Warri, Nigeria','Suite A16 Mc Lewis Plaza IBB Way, Wuse II, Fct, Abuja, Nigeria',
                'Imo State Mini Arts Gallery Block Orlu Road, Secretariat, Imo, Owerri, Nigeria','84A Sanyaolu St. Kano, Nigeria',
                '20, Oremeji Street, Ikeja, Lagos, Nigeria','Shop FB56 First Floor Banex Plaza VOM Aminu Kano Crescent, Wuse II, Fct, Abuja, Nigeria',
                '12 ayo alabi street oke-ira, suite 003-004 oke-ira shopping center ogba , Ikeja, Lagos, Nigeria','Plot 12, Aba Johnson Crescent, Harmony Enclave, Ikeja, Lagos,  Nigeria',
                '10,Sharan Close, Chikun, Kaduna, Nigeria','Amuwo-Odofin, GRA, Festac, Lagos, Lagos, Nigeria',
                '1, Femi Taiwo Close,  Magbon Abeokuta, Abeokuta South, Nigeria','1, Sokoto Road, Zaria, Kaduna, Nigeria',
                '3, Imam Ligali Street, Kosofe, Lagos, Nigeria','A.E. Ekukinam Street, District, Utako, Fct, Abuja, Nigeria',
                'Victoria Island, Lagos, Nigeria','257, Bassan Plaza Central area, Lugbe, Abuja, Nigeria',
                '3, Opebi Road, Ikeja, Lagos, Lagos, Nigeria','Buffalo House, Allen Avenue, Opebi, Ikeja, Lagos, Nigeria',
                '4B, Aggrey Rd, P/h, P/H, Nigeria'))

                api_key = st.text_input("Enter your API key: eg, AIzaSyCpF1uPQLSCrx2CdsLLF_db")

                try:
                    if st.button("Click to Download images"):
                        file_delete(path)
                        ups_download(Address, api_key)
                        _, _, files = next(os.walk("/Users/CHIBUIKEM EFUGHA/OneDrive/Documents/streamlit object detection/Content"))
                        file_count = len(files)
                        st.success(f'{file_count} images downloaded successfully from the address provided')
                    
                    if st.button("Predict"):

                        predicted_images_with_pole, total_images_folder, data, df_load = Detection_downloaded(downloaded_image_path, region_name = ups_download(Address, api_key = "AIzaSyCpF1uPQLSCrx2CdsLLF_dbaZVEQDC1TE4"))

                        st.success(f'There are { predicted_images_with_pole} images with poles out of {total_images_folder} downloaded images')
                        st.success(f'Statistics: {round((predicted_images_with_pole/total_images_folder)*100, 2)} % of the images have pole')
                    
                        data_frame = data.to_csv(index=False).encode('utf-8')
                        
                        # csv = pd.read_csv('data_frame.csv')
                        fig = px.scatter_mapbox(df_load, lat = "Latitude", lon = "Longitude", color="Fibre_Pole_Present", zoom=3, mapbox_style='open-street-map')
                        st.success(fig.show())
                        st.download_button("Click to Download csv", data_frame, "region_name.csv", "text/csv", key='download-csv')

                except KeyboardInterrupt:
                    pass 


            elif input_type == 'Input Region':
                Address = st.text_input('Type in address: ','E.g; Victoria Island, Lagos, Nigeria')
                api_key = st.text_input("Enter your API key: eg, AIzaSyCpF1uPQLSCrx2CdsLLF_db")
                
            
                try:
                    if st.button("Click to Download images"):
                        file_delete(path)
                        ups_download(Address, api_key)
                        _, _, files = next(os.walk("/Users/CHIBUIKEM EFUGHA/OneDrive/Documents/streamlit object detection/Content"))
                        file_count = len(files)
                        st.success(f'{file_count} images downloaded successfully from the address provided')
                    if st.button("Process"):

                        predicted_images_with_pole, total_images_folder, data, df_load = Detection_downloaded(downloaded_image_path, region_name = ups_download(Address, api_key = "AIzaSyCpF1uPQLSCrx2CdsLLF_dbaZVEQDC1TE4"))

                        st.success(f'There are { predicted_images_with_pole} images with poles out of {total_images_folder} downloaded images')
                        st.success(f'Statistics: {round((predicted_images_with_pole/total_images_folder)*100, 2)} % of the images have pole')
                    
                        data_frame = data.to_csv(index=False).encode('utf-8')
                        
                        # csv = pd.read_csv('data_frame.csv')
                        fig = px.scatter_mapbox(df_load, lat = "Latitude", lon = "Longitude", color="Fibre_Pole_Present", zoom=3, mapbox_style='open-street-map')
                        st.success(fig.show())
                        st.download_button("Click to Download csv", data_frame, "region_name.csv", "text/csv", key='download-csv')

                except KeyboardInterrupt:
                    pass 
   



    if selection == "About Us":
        menu = ['Documentation', 'About Team']
        type = st.sidebar.selectbox("Choose what you want to learn about 👇", menu, )

        if type == 'Documentation':
            st.subheader("*App Documentation*: Learn How to use the Fibre Pole Detection App")
            # time.sleep(3)
            # st.button("Go to next page")
            st.write("""
                    This app was primarily created for Fibre Utility Pole Detectiont. There are three pages in the app which 
                                includes; `home page`, `Task page`, and `About`.
                    - *`Home`*: The home page is the app's landing page and includes a welcome message and a succinct summary of the app.
                    
                    - *`Task`*: The Task section is futher divided into two pages; `Utility Pole detection (Image) page`, `Utility Pole detection (google street) page`, 
                    These pages have similar workflow except that the Utility Pole detection(Image) page is used to detect poles on images uploaded and only takes in one image at a time.
                     On the other hand, the Utility Pole detection(google street) page is used to download multiple images from the address provided and detects if poles are available in the provided address,
                      in turn gives you the summary statistics of the street or region in a csv file format which can be downloaded. Instructions and how to input address is provided in the different pages.""")
                    
            
            st.write("""
                    - *`About`*: The About page also has two sub-pages;  `Documentation` and `About Team` page.
                   
                    - *`Documentation`*: This is the current page. It includes a detailed explanation of the app as well as usage guidelines on
                            how to use this app with ease.
                    
                    - *`About Team`*: This page gives you a brief summary of the experience of the team who built and manages the app.
                    """)
        
        else:
            st.subheader("About Team")
            st.markdown("Here are the members Team-12 of the Explore AI Interns who worked and developed this App 👇")
            
            
            Osi, text1 = st.columns((1,2))
        
            with Osi:
                st.image("https://media.discordapp.net/attachments/1009195330354548767/1032761067580956732/Osi.jpg", width = 200)
            

            with text1:
                st.write("""
                    Osigbemhe is  A graduate of Electrical/Electronic Engineering now seeking to leverage his Engineering skills in the field of Data. 
                    I am focused on developing the most efficient methods of data collection and pre-processing to make quality data available for everyday use.
                     A certified AWS Cloud Practitioner and a Fintech Professional. 
                    """)
            st.subheader("Osigbemhe [Team Leader]")
            

            Godwin, text2 = st.columns((1,2))
            
            with Godwin:
                 st.image("https://media.discordapp.net/attachments/1009195330354548767/1032688156803334184/Godwin.jpg", width = 200)

            with text2:
                st.write("""Godwin C. Efugha is a Chemical Engineer with also a degree in Human Anatomy. He is currently a Data Science intern at Explore AI with a background in business analytics, 
                technical writing, deep learning and python programming. Before his journey as a data scientist, he was a freelancer. 
                He is driven by the motto "To help humanity by solving world problems using AI tools"
                    """)
            st.subheader("Efugha C. Godwin [Asst. Team Lead]")
            
            
            Micheal, text3 = st.columns((1, 2))
            
            with Micheal:
                st.image("https://media.discordapp.net/attachments/1009195330354548767/1032669245600964689/Micar.jpg?width=932&height=1007", width = 200)

            with text3:
                st.write("""
                Michael Kihara is a Math graduate. He is currently in data engineer intern at ExploreAI with a background in data science, python programming, 
                content creation, and digital marketing. For the most part, he worked as a freelancer before starting his journey in data science. 
                Michael is currently pursuing a AWS solutions architect certification. 
                    """)
            st.subheader("Micheal Kihara [Team Admin]")

            
            Humphery, text4 = st.columns((1,2))
                    

            with Humphery:
                st.image("https://media.discordapp.net/attachments/1009195330354548767/1032687247331438612/Humphery.jpg?width=803&height=1006", width = 200)

            with text4:
                st.write("""
                Humphery (Osas) Ojo,  an enthusiastic Data Scientist with great euphoria for Exploratory Data Analysis
                (Power-BI, Tableau, Excel, SQL, Python, R) and Machine Learning Engineering(Supervised and Unsupervised Learning), 
                mid-level proficiency in Front-End Web Development(HTML, CSS, MVC, RAZOR, C#).
                """)
            st.subheader("Humphery Ojo [Technical Lead]")


            
            Basheer, text5 = st.columns((1,2))
            
            
            with Basheer:
                st.image("https://media.discordapp.net/attachments/1009195330354548767/1032673227413139576/Basheer.jpg?width=719&height=1006", width = 200)

            with text5:
                st.write("""
                Basheer Ashafa is a graduate of microbiology from the Lagos state university Nigeria. He is currently working as a data science intern at explore Ai . 
                He passionate about the world of Artificial  intelligence and it's application in our modern day society
                    """)
            st.header("Basheer Ashafa  [Asst. Technical Lead]")

        

if __name__ == '__main__':
                main()	