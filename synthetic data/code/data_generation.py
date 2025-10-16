import pandas as pd
from faker import Faker
import random
from datetime import datetime, timedelta
import os

def generate_crm_data():
    """
    Generates a complete, interconnected, and logically sound CRM dataset
    for a real estate developer.
    """
    # --- 1. SETUP AND INITIALIZATION ---
    print("Initializing data generation...")
    fake = Faker('en_IN')  # Use Indian locale for names and addresses

    # Define brokers
    brokers = [
        {"BrokerID": f"BRK{i:03d}", "BrokerName": fake.name()} for i in range(1, 16)
    ]
    broker_ids = [b['BrokerID'] for b in brokers]

    # --- 2. DEFINE THE CORE PROJECT DATA (THE FOUNDATION) ---
    completed_projects = [
        {'ProjectID': 'PROJ001', 'ProjectName': 'Serenity Gardens', 'Location': 'Pune - Hinjawadi', 'LaunchDate': datetime(2010, 3, 15), 'CompletionDate': datetime(2013, 5, 20), 'TotalUnits': 150, 'PriceRange(INR)': '45L - 75L'},
        {'ProjectID': 'PROJ002', 'ProjectName': 'Royal Palms', 'Location': 'Mumbai - Bandra', 'LaunchDate': datetime(2011, 6, 10), 'CompletionDate': datetime(2014, 8, 15), 'TotalUnits': 120, 'PriceRange(INR)': '1.5Cr - 3.0Cr'},
        {'ProjectID': 'PROJ003', 'ProjectName': 'Cyber Vista', 'Location': 'Bengaluru - Whitefield', 'LaunchDate': datetime(2012, 9, 1), 'CompletionDate': datetime(2015, 11, 25), 'TotalUnits': 200, 'PriceRange(INR)': '60L - 1.1Cr'},
        {'ProjectID': 'PROJ004', 'ProjectName': 'Greenwood Estates', 'Location': 'Delhi - Saket', 'LaunchDate': datetime(2014, 2, 20), 'CompletionDate': datetime(2017, 4, 30), 'TotalUnits': 180, 'PriceRange(INR)': '1.2Cr - 2.5Cr'},
        {'ProjectID': 'PROJ005', 'ProjectName': 'Orchid Towers', 'Location': 'Chennai - Adyar', 'LaunchDate': datetime(2015, 11, 12), 'CompletionDate': datetime(2018, 12, 18), 'TotalUnits': 160, 'PriceRange(INR)': '80L - 1.5Cr'},
        {'ProjectID': 'PROJ006', 'ProjectName': 'Capital Heights', 'Location': 'Hyderabad - Gachibowli', 'LaunchDate': datetime(2017, 4, 5), 'CompletionDate': datetime(2020, 7, 22), 'TotalUnits': 250, 'PriceRange(INR)': '70L - 1.3Cr'},
        {'ProjectID': 'PROJ007', 'ProjectName': 'Sapphire Bay', 'Location': 'Goa - Panaji', 'LaunchDate': datetime(2018, 8, 25), 'CompletionDate': datetime(2021, 9, 10), 'TotalUnits': 100, 'PriceRange(INR)': '90L - 2.0Cr'},
        {'ProjectID': 'PROJ008', 'ProjectName': 'The Summit', 'Location': 'Gurgaon - Golf Course Road', 'LaunchDate': datetime(2019, 1, 30), 'CompletionDate': datetime(2022, 3, 5), 'TotalUnits': 220, 'PriceRange(INR)': '2.0Cr - 4.5Cr'},
        {'ProjectID': 'PROJ009', 'ProjectName': 'Emerald Enclave', 'Location': 'Kolkata - Alipore', 'LaunchDate': datetime(2021, 5, 18), 'CompletionDate': datetime(2024, 6, 28), 'TotalUnits': 140, 'PriceRange(INR)': '1.0Cr - 2.2Cr'},
    ]

    ongoing_projects = [
        {'ProjectID': 'PROJ010', 'ProjectName': 'Phoenix Rise', 'Location': 'Pune - Kharadi', 'LaunchDate': datetime(2022, 8, 10), 'ExpectedCompletionDate': datetime(2026, 3, 15), 'TotalUnits': 300, 'UnitsBooked': 110, 'CurrentStatus': 'Construction in full swing; 15 floors completed', 'PercentageCompletion': 65},
        {'ProjectID': 'PROJ011', 'ProjectName': 'Aqua Front', 'Location': 'Mumbai - Navi Mumbai', 'LaunchDate': datetime(2023, 4, 20), 'ExpectedCompletionDate': datetime(2027, 6, 30), 'TotalUnits': 250, 'UnitsBooked': 85, 'CurrentStatus': 'Excavation and foundation work complete; Superstructure work started', 'PercentageCompletion': 40},
        {'ProjectID': 'PROJ012', 'ProjectName': 'Tech Park One', 'Location': 'Bengaluru - Electronic City', 'LaunchDate': datetime(2024, 1, 15), 'ExpectedCompletionDate': datetime(2027, 12, 20), 'TotalUnits': 400, 'UnitsBooked': 120, 'CurrentStatus': 'Plinth level construction underway', 'PercentageCompletion': 25},
    ]

    upcoming_projects = [
        {'ProjectID': 'PROJ013', 'ProjectName': 'Ivory Shore', 'Location': 'Chennai - ECR', 'ProposedLaunchDate': datetime(2026, 6, 1), 'ProposedTimeline': '48 months', 'ProjectScope': 'Ultra-luxury beachfront villas with private pools and smart home automation.', 'Status': 'Awaiting RERA Approval'},
        {'ProjectID': 'PROJ014', 'ProjectName': 'The Citadel', 'Location': 'Hyderabad - Financial District', 'ProposedLaunchDate': datetime(2026, 9, 15), 'ProposedTimeline': '36 months', 'ProjectScope': 'Integrated township with premium 3 & 4 BHK apartments, a commercial complex, and a school.', 'Status': 'Land Acquisition Complete'},
        {'ProjectID': 'PROJ015', 'ProjectName': 'North Star', 'Location': 'Gurgaon - Dwarka Expressway', 'ProposedLaunchDate': datetime(2027, 2, 1), 'ProposedTimeline': '42 months', 'ProjectScope': 'High-rise residential towers featuring smart, sustainable living concepts and sky gardens.', 'Status': 'Planning and Design Stage'},
    ]

    # --- 3. GENERATE SALES EVENTS AND SORT CHRONOLOGICALLY ---
    print("Generating all sale events...")
    all_sales_events = []
    
    # Calculate UnitsSold for completed projects (90%)
    for p in completed_projects:
        p['UnitsSold'] = int(p['TotalUnits'] * 0.9)
        
        property_ids = [f"{p['ProjectID'][-3:]}-{chr(65+i%4)}-{(i//4)+101}" for i in range(p['TotalUnits'])]
        random.shuffle(property_ids)
        
        for i in range(p['UnitsSold']):
            project_duration_days = (p['CompletionDate'] - p['LaunchDate']).days
            random_days = random.randint(30, project_duration_days + 90)
            sale_date = p['LaunchDate'] + timedelta(days=random_days)
            
            # ===================================================================
            # !! BUG FIX !! This new block correctly parses mixed-unit price ranges.
            # ===================================================================
            def parse_price(price_str):
                price_str = price_str.strip()
                if 'Cr' in price_str:
                    return float(price_str.replace('Cr', '')) * 100
                elif 'L' in price_str:
                    return float(price_str.replace('L', ''))
                return 0

            low_str, high_str = p['PriceRange(INR)'].split(' - ')
            low_lakhs = parse_price(low_str)
            high_lakhs = parse_price(high_str)
            sale_price = int(random.uniform(low_lakhs, high_lakhs) * 100000)
            # ===================================================================
            
            all_sales_events.append({
                "DateOfSale": sale_date,
                "ProjectID": p['ProjectID'],
                "ProjectCompletionDate": p['CompletionDate'],
                "PropertyID": property_ids.pop(),
                "SalePrice(INR)": sale_price,
                "BrokerID": random.choice(broker_ids)
            })

    print("Sorting sales chronologically to ensure logical ID assignment...")
    all_sales_events.sort(key=lambda x: x['DateOfSale'])
    
    # --- 4. CREATE FINAL DATAFRAMES FROM SORTED SALES EVENTS ---
    print("Generating final interconnected records from sorted data...")
    sales_records = []
    customer_records = []
    feedback_records = []
    
    feedback_id_counter = 1
    
    for i, event in enumerate(all_sales_events):
        customer_id = f"CUST{i+1:04d}"
        sales_id = f"SALE{i+1:04d}"
        customer_name = fake.name()
        
        sales_records.append({
            "SalesID": sales_id,
            "CustomerID": customer_id,
            "ProjectID": event['ProjectID'],
            "PropertyID": event['PropertyID'],
            "DateOfSale": event['DateOfSale'].strftime('%Y-%m-%d'),
            "SalePrice(INR)": event['SalePrice(INR)'],
            "BrokerID": event['BrokerID'],
            "Status": "Sold"
        })
        
        customer_records.append({
            "CustomerID": customer_id,
            "CustomerName": customer_name,
            "ContactInfo": fake.email(),
            "Address": f"{event['PropertyID']}, {event['ProjectID']}",
            "FirstPurchaseDate": event['DateOfSale'].strftime('%Y-%m-%d')
        })
        
        if random.random() < 0.7:
            feedback_date = event['ProjectCompletionDate'] + timedelta(days=random.randint(30, 365))
            rating = random.randint(2, 5)
            
            if rating == 5:
                comment = random.choice([
                    "Excellent construction quality and timely delivery. Very happy!",
                    "Truly a landmark project. The finish and attention to detail are superb.",
                ])
            elif rating == 4:
                comment = random.choice([
                    "Good project overall, but the clubhouse amenities were delayed.",
                    "Loved the apartment view, though the initial paperwork process was a bit slow.",
                ])
            elif rating == 3:
                comment = random.choice([
                    "Average experience. Construction is solid, but there were some plumbing issues after moving in.",
                    "The apartment is nice, but maintenance fees are higher than what was initially promised."
                ])
            else:
                comment = random.choice([
                    "Significant delay in handover. The final product feels rushed.",
                    "Build quality is not up to the mark. Facing issues with fittings."
                ])

            feedback_records.append({
                "FeedbackID": f"FDBK{feedback_id_counter:04d}",
                "CustomerID": customer_id,
                "ProjectID": event['ProjectID'],
                "FeedbackDate": feedback_date.strftime('%Y-%m-%d'),
                "Rating": rating,
                "Comments": comment
            })
            feedback_id_counter += 1

    df_completed_projects = pd.DataFrame(completed_projects)
    df_ongoing_projects = pd.DataFrame(ongoing_projects)
    df_upcoming_projects = pd.DataFrame(upcoming_projects)
    df_historical_sales = pd.DataFrame(sales_records)
    df_past_customers = pd.DataFrame(customer_records)
    df_feedback = pd.DataFrame(feedback_records)
    
    for df in [df_completed_projects, df_ongoing_projects]:
        for col in ['LaunchDate', 'CompletionDate', 'ExpectedCompletionDate']:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
    if 'ProposedLaunchDate' in df_upcoming_projects.columns:
        df_upcoming_projects['ProposedLaunchDate'] = pd.to_datetime(df_upcoming_projects['ProposedLaunchDate']).dt.strftime('%Y-%m-%d')

    print("Data generation complete.")
    
    return {
        "Completed_Projects": df_completed_projects,
        "Historical_Sales": df_historical_sales,
        "Past_Customers": df_past_customers,
        "Feedback": df_feedback,
        "Ongoing_Projects": df_ongoing_projects,
        "Upcoming_Projects": df_upcoming_projects
    }


if __name__ == "__main__":
    all_dataframes = generate_crm_data()
    
    output_dir = "CRM_Data_Output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for name, df in all_dataframes.items():
        file_path = os.path.join(output_dir, f"{name}.csv")
        df.to_csv(file_path, index=False)
        print(f"Successfully saved {name}.csv to {output_dir}/")