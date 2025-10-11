"""Time parsing module"""

import re
from datetime import datetime, timedelta
from typing import Tuple
import calendar


class TimeParser:
    """Parse various time input formats"""
    
    @staticmethod
    def parse(time_input: str) -> Tuple[datetime, datetime]:
        """Parse time input and return start/end datetime"""
        time_input = time_input.strip()
        
        if re.match(r'^\d{4}-\d{4}$', time_input):
            return TimeParser._parse_year_range(time_input)
        elif re.match(r'^\d{4}$', time_input):
            return TimeParser._parse_year(time_input)
        elif re.match(r'^\d{4}-\d{2}$', time_input):
            return TimeParser._parse_month(time_input)
        elif re.match(r'^\d{4}-\d{2}-\d{2}$', time_input):
            return TimeParser._parse_day(time_input)
        else:
            raise ValueError(
                f"Unsupported time format: {time_input}\n"
                f"Supported formats:\n"
                f"  - Year range: 2020-2023\n"
                f"  - Full year: 2023\n"
                f"  - Full month: 2023-06\n"
                f"  - Single day: 2023-06-15"
            )
    
    @staticmethod
    def _parse_year_range(time_input: str) -> Tuple[datetime, datetime]:
        """Parse year range (e.g., 2020-2023)"""
        start_year, end_year = map(int, time_input.split('-'))
        
        if start_year > end_year:
            raise ValueError(f"Start year {start_year} cannot be greater than end year {end_year}")
        
        start_dt = datetime(start_year, 1, 1, 0, 0, 0)
        end_dt = datetime(end_year, 12, 31, 23, 59, 59)
        
        return start_dt, end_dt
    
    @staticmethod
    def _parse_year(time_input: str) -> Tuple[datetime, datetime]:
        """Parse full year (e.g., 2023)"""
        year = int(time_input)
        start_dt = datetime(year, 1, 1, 0, 0, 0)
        end_dt = datetime(year, 12, 31, 23, 59, 59)
        return start_dt, end_dt
    
    @staticmethod
    def _parse_month(time_input: str) -> Tuple[datetime, datetime]:
        """Parse full month (e.g., 2023-06)"""
        year, month = map(int, time_input.split('-'))
        
        if not (1 <= month <= 12):
            raise ValueError(f"Month must be between 1-12, got: {month}")
        
        start_dt = datetime(year, month, 1, 0, 0, 0)
        last_day = calendar.monthrange(year, month)[1]
        end_dt = datetime(year, month, last_day, 23, 59, 59)
        
        return start_dt, end_dt
    
    @staticmethod
    def _parse_day(time_input: str) -> Tuple[datetime, datetime]:
        """Parse single day (e.g., 2023-06-15)"""
        try:
            dt = datetime.strptime(time_input, '%Y-%m-%d')
        except ValueError as e:
            raise ValueError(f"Invalid date format: {time_input}, error: {e}")
        
        start_dt = dt.replace(hour=0, minute=0, second=0)
        end_dt = start_dt + timedelta(days=1) - timedelta(seconds=1)
        
        return start_dt, end_dt
    
    @staticmethod
    def format_datetime(dt: datetime, format: str = '%Y-%m-%d') -> str:
        """Format datetime to string"""
        return dt.strftime(format)
    
    @staticmethod
    def get_time_range_label(start_dt: datetime, end_dt: datetime) -> str:
        """Generate time range label for filenames"""
        if start_dt.date() == end_dt.date():
            return start_dt.strftime('%Y%m%d')
        elif (start_dt.day == 1 and end_dt.day == calendar.monthrange(end_dt.year, end_dt.month)[1] 
              and start_dt.month == end_dt.month and start_dt.year == end_dt.year):
            return start_dt.strftime('%Y%m')
        elif (start_dt.month == 1 and start_dt.day == 1 
              and end_dt.month == 12 and end_dt.day == 31 
              and start_dt.year == end_dt.year):
            return start_dt.strftime('%Y')
        else:
            return f"{start_dt.strftime('%Y%m%d')}_{end_dt.strftime('%Y%m%d')}"


if __name__ == "__main__":
    parser = TimeParser()
    
    test_cases = ["2020-2023", "2023", "2023-06", "2023-06-15"]
    
    for case in test_cases:
        start, end = parser.parse(case)
        label = parser.get_time_range_label(start, end)
        print(f"{case:15} -> {start} to {end} [label: {label}]")
