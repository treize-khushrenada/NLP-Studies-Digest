from operator import ge
import streamlit as strm
import pandas as pd
import plotly.express as px

@strm.cache
def get_data():
    url = 'http://data.insideairbnb.com/china/hk/hong-kong/2021-03-17/visualisations/listings.csv'
    return pd.read_csv(url)

df_listings = get_data()

strm.title('Hong Kong Airbnb Listings Analysis')
strm.header('Top 10 listings by price')
strm.markdown('last update: march 2021')
strm.markdown('data source: insideairbnb.com')
# strm.dataframe(df_listings.nlargest(10, 'price'))
strm.dataframe(df_listings.loc[:, ['name', 'neighbourhood', 'room_type', 'price']].nlargest(10, 'price'))

strm.header('explore different columns in the dataset')
cols = ["name", "host_name", "neighbourhood", "room_type", "price"]
strm_ms = strm.multiselect("Choose columns: ", df_listings.columns.tolist(), default=cols)
strm.dataframe(df_listings[strm_ms].head())

price_filter = strm.slider('listing areas based on price range', 0, 20000, (0, 1000))
pd.to_numeric(df_listings['price'])

filtered_df_listings = df_listings[df_listings['price'].between(price_filter[0], price_filter[1])]
strm.map(filtered_df_listings)

