#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from datetime import datetime
import os

# Function to save plot
def save_plot(fig, filename):
    """Saves a matplotlib figure as an image file."""
    fig.savefig(filename)
    return filename

# Function to plot data
def plot_data(selected_towns, data):
    if selected_towns:
        filtered_data = data[data['TOWN'].isin(selected_towns)]
    else:
        # If no towns are selected, display data for all towns
        filtered_data = data

    town_dispensing = filtered_data.groupby('TOWN').sum()
    total_dispensed_per_town = town_dispensing.sum(axis=1).sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    total_dispensed_per_town.plot(kind='bar', color='teal')
    plt.title('Total Quantity Sold by Selected Towns')
    plt.xlabel('Town')
    plt.ylabel('Total Quantity Sold')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
    st.pyplot(plt)

# Function to plot data
def plot_product_distribution_by_towns(selected_products, selected_towns, data):
    for selected_product in selected_products:
        # Filtering data based on the selected product and towns
        filtered_data = data[data['DISCRIPTION'] == selected_product]
        if selected_towns:
            filtered_data = filtered_data[filtered_data['TOWN'].isin(selected_towns)]

        town_distribution = filtered_data.groupby('TOWN').sum().sum(axis=1).sort_values(ascending=False)

        plt.figure(figsize=(12, 6))
        town_distribution.plot(kind='bar', color='teal')
        plt.title(f'Distribution of {selected_product} in Selected Towns')
        plt.xlabel('Town')
        plt.ylabel('Total Quantity Sold')
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()
        st.pyplot(plt)

# Streamlit application layout
st.image("logo.png", width=200)
st.title('PCD Sales Analysis Dashboard')

# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.radio('Select an Analysis:', 
                           ['Trend Analysis','Geographical Analysis','Product Performance', 'Pharmacy Performance', 'Alerts'])

# File uploader
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    if options == 'Trend Analysis':
        st.header('Trend Analysis')
        st.subheader('Trend Anaysis Total Units')
        # Include your product performance analysis code here
        # Preparing data for trend analysis
        # Summing up the quantity for each month across all products and pharmacies
             
        # User input for product selection using multiselect
        products = data['DISCRIPTION'].unique().tolist()
        selected_products = st.multiselect('Select Products (leave blank for all products)', products)

        # Filtering data and plotting trends based on product selection
        if selected_products:
            for product in selected_products:
                # Filtering data for the selected product
                filtered_data = data[data['DISCRIPTION'] == product]

                # Preparing data for trend analysis
                monthly_totals = filtered_data.iloc[:, 5:].sum()

                # Plotting the trend for each selected product
                fig, ax = plt.subplots(figsize=(12, 6))
                monthly_totals.plot(kind='bar', color='skyblue', ax=ax)
                plt.title(f'Monthly Sales Trend for {product} (Nov 2022 - Nov 2023)')
                plt.xlabel('Month')
                plt.ylabel('Total Quantity Sold')
                plt.xticks(rotation=45)
                plt.grid(axis='y')

                # Display each plot in Streamlit
                st.pyplot(fig)

                # Save the plot to a BytesIO object and create a download button
                buffer = BytesIO()
                fig.savefig(buffer, format='png')
                buffer.seek(0)
                st.download_button(
                    label=f"Download {product} Trend Image",
                    data=buffer,
                    file_name=f"{product}_trend.png",
                    mime="image/png"
                )
        else:
            # Preparing data for trend analysis for all products
            monthly_totals = data.iloc[:, 5:].sum()

            # Plotting the trend for all products
            fig, ax = plt.subplots(figsize=(12, 6))
            monthly_totals.plot(kind='bar', color='skyblue', ax=ax)
            plt.title('Monthly Sales Trend for All Products (Nov 2022 - Nov 2023)')
            plt.xlabel('Month')
            plt.ylabel('Total Quantity Sold')
            plt.xticks(rotation=45)
            plt.grid(axis='y')

            # Display the plot in Streamlit
            st.pyplot(fig)

            # Save the plot to a BytesIO object and create a download button
            buffer = BytesIO()
            fig.savefig(buffer, format='png')
            buffer.seek(0)
            st.download_button(
                label="Download All Products Trend Image",
                data=buffer,
                file_name="all_products_trend.png",
                mime="image/png"
            )
        

        # Analyzing the trend for each product brand (DISCRIPTION)

        # Grouping the data by 'DISCRIPTION' and summing up the quantities for each month
        product_trends = data.groupby('DISCRIPTION').sum()

        # Transposing the dataframe for easier plotting
        product_trends_transposed = product_trends.T

        # Plotting the trends for the top 6 products based on total quantity dispensed
        st.title("Product Sales Trend Analysis")
        st.subheader('Trend Analysis for Selected Products')

        # User input for selecting products using multiselect
        all_products = product_trends.index.tolist()
        selected_products = st.multiselect('Select Products (leave blank for top 6 products)', all_products)

        # Create a matplotlib figure
        fig, ax = plt.subplots(figsize=(15, 8))

        # Determine the products to plot
        if not selected_products:
            st.subheader('Trend Analysis for Top 6 Products')
            products_to_plot = product_trends.sum(axis=1).nlargest(6).index
            plot_title = 'Monthly Sales Trend for Top 6 Products (Nov 2022 - Nov 2023)'
        else:
            products_to_plot = selected_products
            plot_title = 'Monthly Sales Trend for Selected Products (Nov 2022 - Nov 2023)'

        # Plotting the trend
        for product in products_to_plot:
            product_trends_transposed[product].plot(kind='line', label=product, ax=ax)
        plt.title(plot_title)
        plt.xlabel('Month')
        plt.ylabel('Quantity Sold')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)

        # Display the plot in Streamlit
        st.pyplot(fig)

        # Save the plot to a BytesIO object and create a download button
        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        st.download_button(
            label="Download Trend Image",
            data=buffer,
            file_name="trend_analysis.png",
            mime="image/png"
        )

     
        # User input for number of top products
        st.subheader('Trend Anaysis Top N Products')
        
        num_products = st.number_input('Select number of top products to analyze', min_value=1, value=5, step=1)
        
        # Plotting the trends for the top N products based on total quantity dispensed
        top_products = product_trends.sum(axis=1).nlargest(num_products).index
        fig, ax = plt.subplots(figsize=(15, 8))
        for product in top_products:
            product_trends_transposed[product].plot(kind='line', label=product, ax=ax)
        plt.title(f'Monthly Sales Trend for Top {num_products} Products (Nov 2022 - Nov 2023)')
        plt.xlabel('Month')
        plt.ylabel('Quantity Sold')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.grid(axis='y')

        # Display the plot in Streamlit
        st.pyplot(fig)

        # Save the plot to a BytesIO object and create a download button
        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        st.download_button(
            label="Download Trend Image",
            data=buffer,
            file_name=f"Top_{num_products}_Products_Trend.png",
            mime="image/png"
        )
        
       
        # User input for number of product/products
        st.subheader('Trend Analysis for Selected Product(s)')
        
        # User input for selecting products
        all_products = product_trends_transposed.columns.tolist()
        selected_products = st.multiselect('Select Products', all_products, default=all_products[0])

        # Create a matplotlib figure for the plot
        fig, ax = plt.subplots(figsize=(15, 8))

        # Plotting the trends for the selected products
        for product in selected_products:
            product_trends_transposed[product].plot(kind='line', label=product, ax=ax)
        plt.title('Monthly Sales Trend for Selected Products (Nov 2022 - Nov 2023)')
        plt.xlabel('Month')
        plt.ylabel('Quantity Sold')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.grid(axis='y')

        # Display the plot in Streamlit
        st.pyplot(fig)

        # Save the plot to a BytesIO object and create a download button
        buffer = BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        st.download_button(
            label="Download Trend Image",
            data=buffer,
            file_name="selected_products_trend.png",
            mime="image/png"
        )

    elif options == 'Geographical Analysis':
        st.header('Geographical Analysis')
        st.subheader('Overall Unit Sales by Town')
        
        # Town selection
        towns = data['TOWN'].unique()
        selected_towns = st.multiselect('Select Towns', towns)

        # Display plot
        plot_data(selected_towns, data)
              
        # Town and Product selection
        st.subheader('Unit Sales by Town and Product')
        towns = data['TOWN'].unique()
        products = data['DISCRIPTION'].unique()
        selected_towns = st.multiselect('Select Towns', towns, key='select_towns')
        selected_products = st.multiselect('Select Products', products, key='select_products')

        # Display plot
        if selected_products:
            plot_product_distribution_by_towns(selected_products, selected_towns, data)
        else:
            st.write("Please select one or more products to view their distribution.")
               
    elif options == 'Product Performance':
        st.header('Product Performance')
        
        # Recreating the total_dispensed_per_town variable
        st.header('Total sold per town')
        
        # User input for number of top towns and products
        num_towns = st.number_input('Select number of top towns to analyze', min_value=1, value=10, step=1)
        num_products = st.number_input('Select number of top products to analyze', min_value=1, value=10, step=1)

        # Grouping and summarizing data
        town_dispensing = data.groupby('TOWN').sum(numeric_only=True)
        product_trends = data.groupby('DISCRIPTION').sum(numeric_only=True)
        total_dispensed_per_town = town_dispensing.sum(axis=1, numeric_only=True).sort_values(ascending=False)

        # Selecting top N towns and products
        top_towns = total_dispensed_per_town.nlargest(num_towns).index
        top_products = product_trends.sum(axis=1, numeric_only=True).nlargest(num_products).index

        # Filtering and aggregating data
        filtered_data = data[data['TOWN'].isin(top_towns) & data['DISCRIPTION'].isin(top_products)]
        town_brand_aggregated = filtered_data.groupby(['TOWN', 'DISCRIPTION']).sum(numeric_only=True).reset_index()

        # Creating summary tables for each town and saving them as CSV
        for town in top_towns:
            town_data = town_brand_aggregated[town_brand_aggregated['TOWN'] == town]

            # Adjusting the slicing to include all relevant monthly columns
            monthly_columns = town_data.columns[0:]  # Modify this according to your DataFrame
            summary = town_data.pivot(index='DISCRIPTION', columns='TOWN', values=monthly_columns.tolist()).fillna(0)

            # Displaying the summary table for the town
            st.subheader(f"Summary for {town}:")
            st.table(summary)

            # Converting the DataFrame to CSV and encoding to bytes
            csv = summary.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=f"Download data for {town}",
                data=csv,
                file_name=f"{town}_summary.csv",
                mime='text/csv'
            )
        
        # Calculating the total quantity dispensed for each product
        total_quantity_by_product = data.groupby('DISCRIPTION').sum(numeric_only=True).sum(axis=1).sort_values(ascending=False)
        st.subheader("Total Quantity Sold for Each Product:")
        st.table(total_quantity_by_product.head(10))  # Displaying top 10
        
        # Identifying the top 10 products based on total quantity dispensed
        top_10_products = total_quantity_by_product.nlargest(10)

        # Analyzing monthly trends for these top 10 products
        monthly_trends_top_10 = data[data['DISCRIPTION'].isin(top_10_products.index)].groupby('DISCRIPTION').sum(numeric_only=True)
        st.subheader("Monthly Trends for Top 10 Products:")
        st.table(monthly_trends_top_10)

        # Converting the DataFrame to CSV and encoding to bytes
        csv = monthly_trends_top_10.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="Download Monthly Trends CSV",
            data=csv,
            file_name="monthly_trends_top_10.csv",
            mime='text/csv'
        )

        # Analyzing distribution across towns for these top 10 products
        distribution_across_towns_top_10 = data[data['DISCRIPTION'].isin(top_10_products.index)].groupby(['DISCRIPTION', 'TOWN']).sum(numeric_only=True).sum(axis=1).unstack(fill_value=0)
        st.subheader("Distribution Across Towns for Top 10 Products:")
        st.table(distribution_across_towns_top_10)
        
            # Converting the DataFrame to CSV and encoding to bytes
        csv = distribution_across_towns_top_10.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="Download Distribution CSV",
            data=csv,
            file_name="distribution_across_towns_top_10.csv",
            mime='text/csv'
        )
        
        # Product Performance by Pharmacy filtered by month-year

        # Melting the dataset to convert it from wide format to long format
        # This helps in plotting time series data more easily
        # Unique months in the dataset for the dropdown
        # Melting the DataFrame
        df_melted = data.melt(id_vars=['C-CODE', 'NAME', 'TOWN', 'P-CODE', 'DISCRIPTION'], 
                            var_name='Month', value_name='Sales')
        
        st.subheader('Top N Product Performance by Pharmacy')
        
        months = df_melted['Month'].unique()
        selected_month = st.selectbox('Select a Month', months)

        # User input for number of top products
        num_products = st.number_input('Select number of top products', min_value=1, value=5, step=1)

        # Filtering the data based on the selected month
        filtered_data = df_melted[df_melted['Month'] == selected_month]

        # Aggregating sales data by 'NAME' and 'DISCRIPTION'
        aggregated_data = filtered_data.groupby(['NAME', 'DISCRIPTION'])['Sales'].sum().reset_index()

        # Sorting by sales and getting the top N products
        top_n_products = aggregated_data.sort_values(by='Sales', ascending=False).head(num_products)

        # Displaying the aggregated data
        st.table(top_n_products)
        
        # Converting the DataFrame to CSV and encoding to bytes
        csv = top_n_products.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Top N Products CSV",
            data=csv,
            file_name="top_n_products.csv",
            mime='text/csv'
        )
        
        
        # Product Returns
        # User input for the number of top return alerts
        st.subheader("Top Product Returns")

        # User input for town selection
        towns = data['TOWN'].unique().tolist()
        towns.insert(0, 'All Towns')  # Adding an option to select all towns
        selected_town = st.selectbox('Select Town (or choose "All Towns" for all)', towns)

        # Filtering data by selected town if not 'All Towns'
        if selected_town != 'All Towns':
            data = data[data['TOWN'] == selected_town]

        num_alerts = st.number_input('Select number of top return alerts to display', min_value=1, value=20, step=1)

        # Common function to generate alerts
        def generate_alerts(data):
            alerts = []
            grouped_data = data.groupby(['NAME', 'DISCRIPTION'])

            for (pharmacy, product), group in grouped_data:
                monthly_sales = group.iloc[:, 5:]  # includes monthly last column
                last_month_returns = monthly_sales.iloc[:, -1]  # Focusing on the last month

                if last_month_returns.values[0] < 0:
                    returns = abs(last_month_returns.values[0])
                    alert_message = f"Alert: {product} at {pharmacy} - Returns last month: {returns} units"
                    alerts.append((alert_message, pharmacy, product, returns))

            return alerts

        alerts = generate_alerts(data)

        # Handling no alerts case
        if not alerts:
            st.subheader(f"Top {num_alerts} Product Returns Last Month Alerts")
            st.write("No return for the selected period and/or town")
        else:
            # Displaying top N alerts for narration
            st.subheader(f"Top {num_alerts} Product Returns Last Month Alerts")
            for alert in sorted(alerts, key=lambda x: x[3], reverse=True)[:num_alerts]:
                st.write(alert[0])  # Displaying the alert message

            # Creating and displaying DataFrame
            alerts_df = pd.DataFrame(alerts, columns=["Alert", "Pharmacy", "Product", "Returns"])
            top_alerts_df = alerts_df.sort_values(by='Returns', ascending=False).head(num_alerts)
            st.subheader(f"Top {num_alerts} Product Returns Last Month Alerts Table")
            st.table(top_alerts_df.drop(columns="Alert"))

            # Converting the DataFrame to CSV and encoding to bytes
            csv = top_alerts_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Top Product Returns CSV",
                data=csv,
                file_name="top_product_returns_last_month.csv",
                mime='text/csv'
            )

        # Repeat the adjusted logic for the month before last month section
        # ... [Repeat the logic for the month before last month section, similar to the last month section] ...
        # Common function to generate alerts
        def generate_alerts(data):
            alerts = []
            grouped_data = data.groupby(['NAME', 'DISCRIPTION'])

            for (pharmacy, product), group in grouped_data:
                monthly_sales = group.iloc[:, 5:]  # includes monthly last column
                last_month_returns = monthly_sales.iloc[:, -2]  # Focusing on the month beforelast month

                if last_month_returns.values[0] < 0:
                    returns = abs(last_month_returns.values[0])
                    alert_message = f"Alert: {product} at {pharmacy} - Returns month before last month: {returns} units"
                    alerts.append((alert_message, pharmacy, product, returns))

            return alerts

        alerts = generate_alerts(data)

        # Handling no alerts case
        if not alerts:
            st.subheader(f"Top {num_alerts} Product Returns For Month Before Last Month Alerts")           
            st.write("No return for the selected period and/or town")
        else:
            # Displaying top N alerts for narration
            st.subheader(f"Top {num_alerts} Product Returns Month Before Last Month Alerts")
            for alert in sorted(alerts, key=lambda x: x[3], reverse=True)[:num_alerts]:
                st.write(alert[0])  # Displaying the alert message

            # Creating and displaying DataFrame
            alerts_df = pd.DataFrame(alerts, columns=["Alert", "Pharmacy", "Product", "Returns"])
            top_alerts_df = alerts_df.sort_values(by='Returns', ascending=False).head(num_alerts)
            st.subheader(f"Top {num_alerts} Product Returns Month Before Last Month Alerts Table")
            st.table(top_alerts_df.drop(columns="Alert"))

            # Converting the DataFrame to CSV and encoding to bytes
            csv = top_alerts_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Top Product Returns CSV",
                data=csv,
                file_name="top_product_returns_month_before_last_month.csv",
                mime='text/csv'
            )
         
    elif options == 'Pharmacy Performance':
        st.header('Pharmacy Performance')

        # Let the user select the number of top pharmacies to analyze
        num_pharmacies = st.number_input('Select number of top pharmacies to analyze', min_value=1, value=10, step=1)

        unique_products = data['DISCRIPTION'].unique().tolist()
        unique_products.insert(0, 'All Products')
        selected_product = st.selectbox('Select a Product (or choose "All Products" for all)', unique_products)

        if selected_product != 'All Products':
            data = data[data['DISCRIPTION'] == selected_product]

        # Analysis code
        pharmacy_performance = data.groupby('NAME').sum(numeric_only=True)
        total_dispensed_by_pharmacy = pharmacy_performance.sum(axis=1).sort_values(ascending=False)
        top_n_pharmacies = total_dispensed_by_pharmacy.nlargest(num_pharmacies)
        monthly_trends_top_n_pharmacies = data[data['NAME'].isin(top_n_pharmacies.index)].groupby('NAME').sum(numeric_only=True)

        st.subheader(f'Top {num_pharmacies} Pharmacies by Unit Sales for {selected_product}')
        st.table(top_n_pharmacies)
        
        # CSV Download for Unit Sales
        csv_unit_sales = top_n_pharmacies.to_csv().encode('utf-8')
        st.download_button(
            label="Download Top Pharmacies by Unit Sales",
            data=csv_unit_sales,
            file_name="top_pharmacies_unit_sales.csv",
            mime='text/csv'
        )

        st.subheader(f"Top {num_pharmacies} Monthly Trends by Pharmacy for {selected_product}")
        st.table(monthly_trends_top_n_pharmacies)

        # Month Trends for top pharmacies
        csv_monthly_trends = monthly_trends_top_n_pharmacies.to_csv().encode('utf-8')
        st.download_button(
            label="Download Monthly Trends for Top Pharmacies",
            data=csv_monthly_trends,
            file_name="monthly_trends_top_pharmacies.csv",
            mime='text/csv'
        )

        # Top N Pharmacies Data
        top_n_pharmacies_data = data[data['NAME'].isin(top_n_pharmacies.index)]
        top_n_pharmacies_product_performance = top_n_pharmacies_data.groupby(['NAME', 'DISCRIPTION']).sum(numeric_only=True).sum(axis=1).unstack(fill_value=0)

        st.subheader(f'Top {num_pharmacies} Pharmacies by Product')
        st.table(top_n_pharmacies_product_performance)

        # CSV Download for Pharmacies by Product
        csv_pharmacies_product = top_n_pharmacies_product_performance.to_csv().encode('utf-8')
        st.download_button(
            label="Download Top Pharmacies by Product",
            data=csv_pharmacies_product,
            file_name="top_pharmacies_by_product.csv",
            mime='text/csv'
        )
        
        # Pharmacies with at least a product sales
        # Function to process data
        months = data.columns[5:]
        pharmacies_with_sales = data.groupby('NAME').filter(lambda x: all(x[month].sum() > 0 for month in months))

        
        # Calculate total and monthly sales for each of these pharmacies
        total_and_monthly_sales = pharmacies_with_sales.groupby('NAME')[months].sum()
        total_and_monthly_sales['Total Sales'] = total_and_monthly_sales.sum(axis=1)
        
         # User input for top N pharmacies
        top_n = st.number_input('Enter number of Top Pharmacies to display', min_value=1, value=5)

        # Filter by Top N pharmacies based on total sales within the eligible pharmacies
        top_pharmacies = total_and_monthly_sales.sort_values(by='Total Sales', ascending=False).head(top_n)

        # Display the top pharmacies with their monthly and total sales
        st.subheader(f"Top {top_n} Pharmacies by Total Unit Sales (with at least one sale per month)")
        st.write(top_pharmacies)
        
        # Convert the data to a CSV for download
        csv = top_pharmacies.to_csv().encode('utf-8')
        st.download_button(
            label="Download Top Pharmacies Sales Data as CSV",
            data=csv,
            file_name='top_pharmacies_sales_data.csv',
            mime='text/csv',
        )
        
        # Define month columns
        months = data.columns[5:]

        # Identify pharmacies with at least one product sale per month
        pharmacies_with_sales = data.groupby('NAME').filter(lambda x: all(x[month].sum() > 0 for month in months))

        # Calculate total and monthly sales for each of these pharmacies
        total_and_monthly_sales = pharmacies_with_sales.groupby('NAME')[months].sum()
        total_and_monthly_sales['Total Sales'] = total_and_monthly_sales.sum(axis=1)

        # Determine the top N pharmacies based on total sales
        top_pharmacies = total_and_monthly_sales.sort_values(by='Total Sales', ascending=False).head(top_n)

        # Extract product sales data for these top pharmacies
        top_pharmacy_names = top_pharmacies.index.tolist()
        top_pharmacies_product_sales = pharmacies_with_sales[pharmacies_with_sales['NAME'].isin(top_pharmacy_names)]
        st.subheader(f"Top {top_n} Pharmacies by Unit Sales")
        st.write(top_pharmacies_product_sales)
        
        # Convert the data to a CSV for download
        csv = top_pharmacies_product_sales.to_csv().encode('utf-8')
        st.download_button(
            label = "Download Top Pharmacies Unit Sales as CSV",
            data = csv,
            file_name='top_pharmacies_unit_sales.csv',
            mime='text/csv',
        )
        
    elif options == 'Alerts':
        st.header('Pharmacy - Products Alerts')
        
        # User input for town selection
        towns = data['TOWN'].unique().tolist()
        towns.insert(0, 'All Towns')  # Adding an option to select all towns
        selected_town = st.selectbox('Select Town (or choose "All Towns" for all)', towns)

        # Filtering data by selected town if not 'All Towns'
        if selected_town != 'All Towns':
            data = data[data['TOWN'] == selected_town]

        # User input for pharmacy selection
        pharmacies = data['NAME'].unique().tolist()  # Assuming 'NAME' column contains Pharmacy names
        pharmacies.insert(0, 'All Pharmacies')  # Adding an option to select all pharmacies
        selected_pharmacy = st.selectbox('Select Pharmacy (or choose "All Pharmacies" for all)', pharmacies)

        # Filtering data by selected pharmacy if not 'All Pharmacies'
        if selected_pharmacy != 'All Pharmacies':
            data = data[data['NAME'] == selected_pharmacy]
        
        # User input for the number of top alerts
        num_alerts = st.number_input('Select number of top alerts to display', min_value=1, value=20, step=1)

        # Function for generating alerts
        def alert_sales_dip(data, num_alerts, period_desc):
            alerts = []
            grouped_data = data.groupby(['NAME', 'DISCRIPTION'])

            for (pharmacy, product), group in grouped_data:
                monthly_sales = group.iloc[:, 5:] # includes monthly last column
                average_sales = monthly_sales.mean(axis=1)

                last_month_sales = monthly_sales.iloc[:, -1] if period_desc == "the last month" else monthly_sales.iloc[:, -2]
                sales_dip = average_sales.values[0] - last_month_sales.values[0]
                if last_month_sales.values[0] < average_sales.values[0]:
                    sales_dip_rounded = round(sales_dip)
                    alert_msg = f"Alert: {product} at {pharmacy} - Sales in {period_desc} for {selected_town} for {selected_pharmacy} are below the average by {sales_dip_rounded} units"
                    alerts.append((alert_msg, sales_dip_rounded))

            if not alerts:
                return [f"No alerts for {selected_town} in {period_desc}"], []

            # Sorting and limiting alerts
            top_alerts = sorted(alerts, key=lambda x: x[1], reverse=True)[:num_alerts]
            alert_messages = [alert[0] for alert in top_alerts]
            alert_data = [{"Pharmacy": alert[0].split(' at ')[1].split(' - ')[0], "Product": alert[0].split('Alert: ')[1].split(' at ')[0], "Sales Dip": alert[1]} for alert in top_alerts]
            return alert_messages, alert_data

        # Narration and dataframe for last month
        st.header(f"Top {num_alerts} Alerts for {selected_town} for {selected_pharmacy} Narration for last month")
        sales_alerts, alert_data = alert_sales_dip(data, num_alerts, "the last month")
        for alert in sales_alerts:
            st.write(alert)  # Displaying the top N alerts

        alerts_df = pd.DataFrame(alert_data)

        if alerts_df.empty:
            st.write(f"No alerts for {selected_town} in the last month")
        else:
            st.subheader(f"Top {num_alerts} Alerts for {selected_town} for {selected_pharmacy} Table for last month")
            st.table(alerts_df)

            # CSV download button for last month
            csv_last_month = alerts_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Top Alerts for Last Month CSV",
                data=csv_last_month,
                file_name="top_sales_alerts_last_month.csv",
                mime='text/csv'
            )

        # Narration and dataframe for the month before last month
        st.header(f"Top {num_alerts} Alerts for {selected_town} for {selected_pharmacy} Narration for month before last")
        sales_alerts, alert_data = alert_sales_dip(data, num_alerts, "the month before last")
        for alert in sales_alerts:
            st.write(alert)  # Displaying the top N alerts

        alerts_df = pd.DataFrame(alert_data)

        if alerts_df.empty:
            st.write(f"No alerts for {selected_town} in the month before last")
        else:
            st.subheader(f"Top {num_alerts} Alerts for {selected_town} for {selected_pharmacy} Table for month before last")
            st.table(alerts_df)

            # CSV download button for the month before last month
            csv_month_before_last = alerts_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Top Alerts for Month Before Last CSV",
                data=csv_month_before_last,
                file_name="top_sales_alerts_month_before_last.csv",
                mime='text/csv'
            )
        
        # Data validation for product alets
        # Load data frame
        df = pd.DataFrame(data)

        # Streamlit UI
        st.subheader('Pharmacy Product Sales Data')

        # Dropdown for selecting Pharmacy
        pharmacy_list = df['NAME'].unique().tolist()
        selected_pharmacy = st.selectbox('Select Pharmacy', pharmacy_list)

        # Multiselect for selecting Products
        product_list = ['All Products'] + df['DISCRIPTION'].unique().tolist()
        # Set default to 'All Products'
        selected_products = st.multiselect('Select Products', product_list, default=['All Products'])

        # Logic to handle the "All Products" selection
        if 'All Products' in selected_products and len(selected_products) > 1:
            selected_products = ['All Products']

        # Filter DataFrame based on selections
        filtered_df = df[df['NAME'] == selected_pharmacy]

        # Check if 'All Products' is selected
        if 'All Products' in selected_products:
            # Do not filter by products
            filtered_df = filtered_df
        else:
            # Filter by selected products
            filtered_df = filtered_df[filtered_df['DISCRIPTION'].isin(selected_products)]

        # Display the filtered DataFrame
        st.write('Filtered Data:', filtered_df)

        # Convert DataFrame to CSV. Index=False to exclude DataFrame index from the CSV
        csv = filtered_df.to_csv(index=False).encode('utf-8')

        # Create a download button and provide the CSV file to download
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='alerts_data.csv',
            mime='text/csv',
        )
        
        # Product sales two or more consecutive drops of a variable or more
        # Streamlit User Interface
        st.subheader("Two or More Consecutive Drop in Unit Sales")
        
        # Ensure tqdm is patched into pandas
        tqdm.pandas()

        # User input for the percentage drop
        percentage_drop = st.number_input("Enter the percentage drop for alert (e.g., 25 for 25%)", min_value=10, value=25, max_value=30)
        drop_threshold = -percentage_drop / 100

        # Convert all sales columns to numeric, coercing errors to NaN
        sales_cols = df.columns[df.columns.str.contains('-')]
        df[sales_cols] = df[sales_cols].apply(pd.to_numeric, errors='coerce')

        # Define a function to check for two or more consecutive drops in sales
        def check_consecutive_drops(row):
            # Calculate the percentage change between months
            changes = row[sales_cols].pct_change().fillna(0)
            # Check for two or more consecutive drops of specified percentage or more
            drops = (changes <= drop_threshold)
            return drops.sum() >= 2

        # Dropdown for product selection
        product_list = df['DISCRIPTION'].unique()
        selected_product = st.selectbox("Select a Product", product_list)

        # Dropdown for town selection
        town_list = df['TOWN'].unique()
        selected_town = st.selectbox("Select a Town", town_list)

        # Filter the DataFrame by 'DISCRIPTION' and 'TOWN'
        filtered_df = df[(df['DISCRIPTION'] == selected_product) & (df['TOWN'] == selected_town)]

        # Apply the function across the rows
        consecutive_drops = filtered_df.progress_apply(check_consecutive_drops, axis=1)

        # Filter the DataFrame based on the condition
        final_filtered_df = filtered_df[consecutive_drops]

        # Drop the 'P-CODE' and 'C-CODE' columns
        final_filtered_df = final_filtered_df.drop(columns=['P-CODE', 'C-CODE'])

        # Display the filtered DataFrame
        st.write(final_filtered_df)

        # Convert DataFrame to CSV. Index=False to exclude DataFrame index from the CSV
        csv = final_filtered_df.to_csv(index=False).encode('utf-8')

        # Create a download button and provide the CSV file to download
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='product_alerts_data.csv',
            mime='text/csv',
        )
else:
    st.warning('Please upload a CSV file to proceed.')



# In[ ]:




