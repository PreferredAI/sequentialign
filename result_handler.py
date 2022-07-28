import numpy as np
import json

def get_gs_timings(dn, deck_no):
    groundstamps = np.loadtxt("./data/" + dn + "/" + str(deck_no) + "/annotation.txt", dtype="str")
    groundstamps = np.delete(groundstamps, [0])

    gs_timings = dict()
    for gs in range(len(groundstamps)):
        gsa = groundstamps[gs].split("-")
        if gsa[1] == '0' and gsa[2] == '0':
            continue
        gs_timings[int(gsa[0])] = (int(gsa[1]), int(gsa[2]))

    return gs_timings

class ResultDump:
    def __init__(self, filename):
        self.filename = filename
        self.gs_timings = {}
        self.srs = {}
        self.srsa = {}
        self.crs = {}
        self.results = {}

    def dump(self):
        # print(self.srs)
        with open('./cache/' + self.filename + '_gs_timings.json', 'w') as fp:
            json.dump(self.gs_timings, fp)
        with open('./cache/' + self.filename + '_srs.json', 'w') as fp:
            json.dump(self.srs, fp)
        with open('./cache/' + self.filename + '_srsa.json', 'w') as fp:
            json.dump(self.srsa, fp)
        with open('./cache/' + self.filename + '_crs.json', 'w') as fp:
            json.dump(self.crs, fp)
        with open('./results/' + self.filename + '_results.json', 'w') as fp:
            json.dump(self.results, fp)

    def load(self):
        with open('./cache/' + self.filename + '_gs_timings.json', encoding='utf-8') as fp:
            self.gs_timings = json.load(fp)
        self.gs_timings = {int(k):v for k,v in self.gs_timings.items()}
        for deck_no in self.gs_timings:
            self.gs_timings[deck_no] = {int(k): v for k, v in self.gs_timings[deck_no].items()}
        with open('./cache/' + self.filename + '_srs.json', encoding='utf-8') as sp:
            self.srs = json.load(sp)
        self.srs = {int(k): v for k, v in self.srs.items()}
        for deck_no in self.srs:
            self.srs[deck_no] = {int(k): v for k, v in self.srs[deck_no].items()}
        with open('./cache/' + self.filename + '_crs.json', encoding='utf-8') as cp:
            self.crs = json.load(cp)
        self.crs = {int(k): v for k, v in self.crs.items()}
        for deck_no in self.crs:
            self.crs[deck_no] = {int(k): v for k, v in self.crs[deck_no].items()}
            for phold in self.crs[deck_no]:
                self.crs[deck_no][phold] = {int(k): v for k, v in self.crs[deck_no][phold].items()}
        with open('./cache/' + self.filename + '_srsa.json', encoding='utf-8') as cpa:
            self.srsa = json.load(cpa)
        self.srsa = {int(k): v for k, v in self.srsa.items()}
        # print(self.srsa)
        for deck_no in self.srsa:
            self.srsa[deck_no] = {int(k): v for k, v in self.srsa[deck_no].items()}
            for phold in self.srsa[deck_no]:
                self.srsa[deck_no][phold] = {int(k): v for k, v in self.srsa[deck_no][phold].items()}
        with open('./results/' + self.filename + '_results.json', encoding='utf-8') as rp:
            self.results = json.load(rp)
        self.results = {int(k): v for k, v in self.results.items()}
        for deck_no in self.results:
            self.results[deck_no] = {int(k): v for k, v in self.results[deck_no].items()}
            for type in self.results[deck_no]:
                for phold in self.results[deck_no][type]:
                    self.results[deck_no][type][phold] = {int(k): v for k, v in self.results[deck_no][type][phold].items()}

    def get_gs_timings(self, deck_no):
        return self.gs_timings[deck_no]

    def get_results(self, deck_no, phold, type):
        return self.results[deck_no][phold][type]

    def get_sr(self, deck_no):
        return self.srs[deck_no]

    def get_sra(self, deck_no, phold):
        return self.srsa[deck_no][phold]

    def get_cr(self, deck_no, phold):
        return self.crs[deck_no][phold]

    def save_gs_timings(self, deck_no, gs_timings):
        self.gs_timings[deck_no] = gs_timings

    def save_sr(self, deck_no, sr):
        self.srs[deck_no] = sr

    def save_sra(self, deck_no, phold, sra):
        if deck_no in self.srsa:
            self.srsa[deck_no][phold] = sra
        else:
            self.srsa[deck_no] = {}
            self.srsa[deck_no][phold] = sra

    def save_cr(self, deck_no, phold, cr):
        if deck_no in self.crs:
            self.crs[deck_no][phold] = cr
        else:
            self.crs[deck_no] = {}
            self.crs[deck_no][phold] = cr

    def save_results(self, deck_no, phold, rtype, result):
        if deck_no in self.results:
            if phold in self.results[deck_no]:
                self.results[deck_no][phold][rtype] = result
            else:
                self.results[deck_no][phold] = {}
                self.results[deck_no][phold][rtype] = result
        else:
            self.results[deck_no] = {}
            self.results[deck_no][phold] = {}
            self.results[deck_no][phold][rtype] = result

    def reset(self, filename):
        self.filename = filename
        self.gs_timings = {}
        self.srs = {}
        self.crs = {}
        self.results = {}

class CategoricalResultHandler:
    def __init__(self, category):
        self.vidaccs = []
        self.mious = []
        self.category = category

    def add_results(self, vacc, miou):
        self.vidaccs.append(vacc)
        self.mious.append(miou)

    def print_avgs(self, key):
        print("\t\t", "Accuracy", "", "IoU")
        print(self.category + "-" + str(key), round(np.mean(self.vidaccs), 3), "\t  ", round(np.mean(self.mious), 3))

    def print_all(self):
        print()
        for i in range(len(self.vidaccs)):
            print(self.category, "\t", i, "\t", self.vidaccs[i], "\t", self.mious[i])
        print(self.category, "\t", np.mean(self.vidaccs), "\t", np.mean(self.mious))

    def reset(self):
        self.vidaccs = []
        self.mious = []

class KeyedCategoricalResultHandler:
    def __init__(self, category):
        self.globadict = {}
        self.category = category

    def add_results(self, vacc, miou, key):
        if key in self.globadict:
            self.globadict[key].add_results(vacc, miou)
        else:
            self.globadict[key] = CategoricalResultHandler(self.category)
            self.globadict[key].add_results(vacc, miou)

    def print_avgs(self, key):
        # print(key, end=" ")
        self.globadict[key].print_avgs(key)

    def reset(self):
        self.globadict = {}