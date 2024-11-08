import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
from typing import List


class MedicalDataSimulator:
    def __init__(self, n_samples: int = 12950):
        self.n_samples = n_samples
        self.stats = {
            'Age': {
                'mean': 35.100764,
                'std': 4.543851,
                'min': 19.0,
                'max': 53.3
            },
            'Weight_kg': {
                'mean': 67.119181,
                'std': 12.550083,
                'min': 32.0,
                'max': 120.0
            }
        }
        self.clinics = ['WESX', 'CMM']
        self.trig_units = ['mcg', 'j.m.', 'IU', 'mg']

    def _generate_follicle_growth(self, start_date: pd.Timestamp,
                                  end_date: pd.Timestamp) -> List:
        """
        Generate realistic follicle growth patterns between dates.
        Returns a list of lists in the exact format: [day, offset, date, follicles, total, count]
        """
        days_between = (end_date - start_date).days
        n_scans = min(days_between, random.randint(1, 4))

        # Initialize follicles
        n_follicles = random.randint(4, 15)
        follicles = [
            {
                'size': random.uniform(4, 8),
                'growth_rate': random.uniform(1.0, 2.0)
            }
            for _ in range(n_follicles)
        ]

        print('days between:', days_between)
        print('n_scans:', n_scans)

        # Generate scan days (0 being start date)
        scan_days = sorted(random.sample(range(3, days_between + 1), n_scans))
        print('scan_days:', scan_days)

        scans = []
        for day in scan_days:
            # Get date for this scan
            scan_date = start_date + pd.Timedelta(days=day)

            # Grow follicles
            current_sizes = []
            for follicle in follicles:
                grown_size = follicle['size'] + (follicle['growth_rate'] * day)
                grown_size += random.uniform(-0.5, 0.5)
                current_sizes.append(round(max(grown_size, 4.0), 1))

            # Sort sizes and filter out small follicles
            current_sizes = sorted([size for size in current_sizes if size >= 5.0])

            if current_sizes:  # Only add scan if there are measurable follicles
                # Format: [day, offset, date, follicles, total_size, count]
                scan = [
                    float(day),  # day number from start
                    -4.0,  # standard offset
                    scan_date.strftime('%Y-%m-%d %H:%M:%S'),
                    current_sizes,
                    round(sum(current_sizes), 2),
                    len(current_sizes)
                ]
                scans.append(scan)

        return scans if scans else [[0.0, -4.0, start_date.strftime('%Y-%m-%d %H:%M:%S'), [5.0], 5.0, 1]]

    def generate_dataset(self) -> pd.DataFrame:
        """Generate complete simulated dataset."""
        data = []

        for i in range(self.n_samples):
            # Generate dates
            treatment_date = pd.Timestamp('2020-01-01') + pd.Timedelta(days=random.randint(0, 365 * 3))
            trigger_date = treatment_date + pd.Timedelta(days=random.randint(10, 14))

            # Generate scans
            tracking_scans = self._generate_follicle_growth(treatment_date, trigger_date)

            # Generate patient metrics
            if random.random() < 0.4:
                age = -100.0
                weight = -1.0
                height = -1.0
                bmi = -1.0
            else:
                age = random.gauss(self.stats['Age']['mean'], self.stats['Age']['std'])
                weight = random.gauss(self.stats['Weight_kg']['mean'], self.stats['Weight_kg']['std'])
                height = round(random.uniform(150, 180), 1)
                bmi = round((weight / (height / 100) ** 2), 2)

            # Generate treatment info
            has_trigger = random.random() < 0.6
            if has_trigger:
                trig_dose = random.choice([250.0, 6500.0, 10000.0])
                trig_units = random.choice(self.trig_units)
                trig_drug = f"Drug_{random.randint(1, 10)}"
            else:
                trig_dose = np.nan
                trig_units = np.nan
                trig_drug = np.nan
                trigger_date = pd.NaT

            row = {
                '': i,
                'Clinic': random.choice(self.clinics),
                'PatientIdentifier': f"CLINIC-{random.randint(100, 999999)}",
                'Cycle Number': f"CYCLE-{random.randint(100, 999999)}",
                'Treatment Start Date': treatment_date,
                'Age at Egg Collection': age,
                'BMI': bmi,
                'Weight_kg': weight,
                'Height_cm': height,
                'Trigger Date': trigger_date,
                'trig_drug': trig_drug,
                'trig_dose': trig_dose,
                'trig_units': trig_units,
                'DoT Follicles': np.nan,
                'Tracking Scans': tracking_scans,  # Now a list of lists, not a string
                'AFC_result': np.nan,
                'amh_value': np.nan,
                'fsh_values': np.nan,
                'lh_values': np.nan
            }

            data.append(row)

        df = pd.DataFrame(data)

        # Ensure dates are datetime
        df['Treatment Start Date'] = pd.to_datetime(df['Treatment Start Date'])
        df['Trigger Date'] = pd.to_datetime(df['Trigger Date'])

        # Calculate Trigger Day
        df['Trigger Day'] = (df['Trigger Date'] - df['Treatment Start Date']).dt.days

        return df