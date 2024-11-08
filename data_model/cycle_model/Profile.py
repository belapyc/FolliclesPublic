from .Bin import Bin, BinType
import warnings


class Profile:
    def __init__(self, follicles, key, day, drug_dosage=None, afc_count=None):
        self.bins = {}
        self.follicles = follicles
        self.follicles.sort()
        self.id = key
        self.day = int(day)
        self.length = len(follicles)
        self.make_bins()
        self.simulation_r2 = None
        self.histosection_score = None
        self.distance = None
        # print('drug dosage: ' + str(drug_dosage))
        self.drug_dosage = drug_dosage
        self.afc_count = afc_count
        self.drug_dosage_per_weight = None

    def make_bins(self, custom_bins=False, bins=None):
        if custom_bins and (bins is None):
            raise Exception('[NOT IMPLEMENTED] Custom bins flag is set to true but no bins were provided.')

        sizeq = int(len(self.follicles) / 4)
        self.bins['top'] = Bin(self.follicles[sizeq * 3:], BinType.TOP_QUARTILE)
        self.bins['upper'] = Bin(self.follicles[sizeq * 2:sizeq * 3], BinType.UPPER_MID_QUARTILE)
        self.bins['lower'] = Bin(self.follicles[sizeq:sizeq * 2], BinType.LOWER_MID_QUARTILE)
        self.bins['bottom'] = Bin(self.follicles[:sizeq], BinType.BOTTOM_QUARTILE)

    def reinit(self, follicles):
        self.follicles = follicles
        self.follicles.sort()
        self.length = len(follicles)

    def __str__(self):
        return str(self.follicles)

    def __eq__(self, other):
        warnings.warn('Comparing profiles is comparing day and follicles. This may not be what you want.')
        return self.day == other.day and \
               self.follicles == other.follicles
