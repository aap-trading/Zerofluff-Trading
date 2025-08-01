
class DaySummary:
    def __init__(self, date, day_high, day_low, or_high, or_low, day_data):
        self.date = date
        self.day_high = day_high
        self.day_low = day_low
        self.or_high = or_high
        self.or_low = or_low
        self.day_data = day_data
        self.prev_day_high = None
        self.prev_day_low = None
        self.prev_VAH = None
        self.prev_VAL = None
        self.vix_open = None