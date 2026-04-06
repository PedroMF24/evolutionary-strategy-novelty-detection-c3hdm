import os

import Higgs.bounds as HB
import Higgs.predictions as HP
import Higgs.signals as HS
import numpy as np
import pandas as pd

pred = HP.Predictions()  # Create the model predictions
pathHB = "/usr/local/lib/hbdataset-v1.6"
pathHS = "/usr/local/lib/hsdataset-v1.1"
bounds = HB.Bounds(pathHB)  # Load HB dataset
signals = HS.Signals(pathHS)  # Load HS dataset

#####################################################################################################################################################
##################### CREATE SCALARS
#####################################################################################################################################################
Nn, Nc = 5, 2
neutral = [
    pred.addParticle(HP.BsmParticle("h" + str(i + 1), "neutral", "undefined"))
    for i in range(Nn)
]
charged = [
    pred.addParticle(HP.BsmParticle("Hp" + str(i + 1), "single", "undefined"))
    for i in range(Nc)
]


#####################################################################################################################################################
##################### COLUMNS READ FROM FORTRAN OUTPUT - File ForHiggsTools.dat (602)
#####################################################################################################################################################
# input variables
columns = [
	'mh1',
	'mh2',
	'mHp1',
	'mHp2',
	'mu12R',
	'mu13R',
	'mu23R',
	'bet1',
	'bet2',
	'tht',
	'pp',
	'a12',
	'a13',
	'a14',
	'a15',
	'a23',
	'a24',
	'a25',
	'a34',
	'a35',
	'a45',
	'mh3',
	'mh4',
	'mh5',
	'mu12I',
    	'L1',
	'L2',
	'L3',
	'L4',
	'L5',
	'L6',
	'L7',
	'L8',
	'L9',
	'L10R',
	'L10I',
	'L11R',
	'L11I',
	'L12R',
	'L12I',
]
# point index
columns.append("index")
# total widths
for i in range(Nn):
    columns.append("Wt_h{}".format(i + 1))
for i in range(Nc):
    columns.append("Wt_Hp{}".format(i + 1))
# couplings hff
for ff in ["uu", "dd", "ee"]:
    for chir in ["s", "p"]:
        for i in range(Nn):
            columns.append("g_h{}{}_{}".format(i + 1, ff, chir))
# couplings hvv
for vv in ["ZZ", "WW", "gaga", "Zga", "gg"]:
    for i in range(Nn):
        columns.append("g_h{}{}".format(i + 1, vv))
# h production xs's
for chanel in ["gg", "VBF", "VH", "ttH", "bbH", "tot"]:
    for i in range(Nn):
        columns.append("xs_h{}_{}".format(i + 1, chanel))
# things for Hp production xs
for i in range(Nc):
    columns.append("etaL_quark_Hp{}".format(i + 1))
for i in range(Nc):
    columns.append("etaR_quark_Hp{}".format(i + 1))
for i in range(Nc):
    columns.append("BR_tHp{}b".format(i + 1))
# BR charged scalars
# for i in range(Nc):
# 	for j in range(Nc):
# 		for k in range(Nn): columns.append('BR_Hp{}Hp{}h{}'.format(i+1,j+1,k+1))
for i in range(Nc):
    for j in range(Nn):
        columns.append("BR_Hp{}h{}W".format(i + 1, j + 1))
for i in range(Nc):
    columns.append("BR_Hp{}WZ".format(i + 1))
for i in range(Nc):
    for f in range(3):
        columns.append("BR_Hp{}LL{}".format(i + 1, f + 1))
for i in range(Nc):
    for f in range(3):
        for j in range(3):
            columns.append("BR_Hp{}QQ{}{}".format(i + 1, f + 1, j + 1))
# BR neutral scalars
for vv in ["WW", "ZZ", "gamgam", "Zgam", "gg"]:
    for j in range(Nn):
        columns.append("BR_h{}{}".format(j + 1, vv))
for ff in ["ee", "mumu", "tautau", "dd", "uu", "ss", "cc", "bb", "tt"]:
    for j in range(Nn):
        columns.append("BR_h{}{}".format(j + 1, ff))
for i in range(Nn):
    for j in range(Nn):
        columns.append("BR_h{}h{}Z".format(i + 1, j + 1))
for i in range(Nn):
    for j in range(Nc):
        columns.append("BR_h{}Hp{}W".format(i + 1, j + 1))
for i in range(Nn):
    for j in range(Nn):
        for k in range(j, Nn):
            columns.append("BR_h{}h{}h{}".format(i + 1, j + 1, k + 1))
for i in range(Nn):
    columns.append("BR_h{}invisible".format(i + 1))


for i in range(Nc):
    for k in range(Nc):
        for j in range(Nn):
            columns.append("BR_Hp{}Hp{}h{}".format(i + 1, k + 1, j + 1))

if 0:
    print()
    print(len(columns))
    print()
    for i, col in enumerate(columns):
        print(i, col)
    print()


#####################################################################################################################################################
##################### ReadFortranOutputFilesToDF with Tests
#####################################################################################################################################################


def ReadFortranOutputFilesToDF(path):
    data = np.genfromtxt(path + "ForHiggsTools.dat")
    df = pd.DataFrame(data, columns=columns)
    df = df.astype({"index": int})

    if 1:  # test some things

        if min(df["mh1"]) < 124 or max(df["mh1"]) > 125.5:
            print("\n * HT WARNING * mh1 != 125\n")

        for col in df.columns:
            if col[:2] == "mh" or col[:3] == "mHp":
                min_mass = min(df[col])
                if min_mass < 0:
                    print("\n *  HT WARNING * negative mass ", min_mass, "\n")
            if col[:3] == "BR_":
                min_br, max_br = min(df[col]), max(df[col])
                if min_br < 0 or max_br > 1:
                    print(
                        "\n * HT WARNING * bad br min max values ", min_br, max_br, "\n"
                    )
            if col[:3] == "Wt_":
                min_w = min(df[col])
                if min_w < 0:
                    print("\n * WARNING * negative total width ", min_w, "\n")

        for particle in ["h{}".format(j + 1) for j in range(Nn)]:
            br_sum = np.zeros(len(df))
            for col in df.columns:
                if col.startswith("BR_{}".format(particle)):
                    if col.endswith("invisible"):
                        continue
                    br_sum += np.array(df[col])
                    if 0:
                        print(np.array(df[col]), col)
            min_br_sum, max_br_sum = min(br_sum), max(br_sum)
            if min_br_sum < 0.99999 or max_br_sum > 1.00001:
                if max_br_sum > 0:
                    print(
                        " * WARNING * {} * br sum not exactly one ".format(particle),
                        br_sum,
                    )

        for particle in ["Hp{}".format(j + 1) for j in range(Nc)]:
            br_sum = np.zeros(len(df))
            for col in df.columns:
                if col.startswith("BR_{}".format(particle)):
                    br_sum += np.array(df[col])
                    # 					print(br_sum)
                    if 0:
                        print(np.array(df[col]), col)
            # 			print(br_sum)
            min_br_sum, max_br_sum = min(br_sum), max(br_sum)
            if min_br_sum < 0.99999 or max_br_sum > 1.00001:
                if max_br_sum > 0:
                    print(
                        " * WARNING * {} * br sum not exactly one ".format(particle),
                        br_sum,
                    )

    return df


#####################################################################################################################################################
##################### Set Scalar Properties
#####################################################################################################################################################


def SetScalarMasses(pt):
    for i in range(Nn):
        neutral[i].setMass(pt["mh{}".format(i + 1)])
    for i in range(Nc):
        charged[i].setMass(pt["mHp{}".format(i + 1)])


def SetScalarTotalWidths(pt):
    for i in range(Nn):
        neutral[i].setTotalWidth(pt["Wt_h{}".format(i + 1)])
    for i in range(Nc):
        charged[i].setTotalWidth(pt["Wt_Hp{}".format(i + 1)])


def SetNeutralScalarCrossSections(pt, flags):
    for i in range(Nn):

        xs_gg = pt["xs_h{}_gg".format(i + 1)]
        xs_bb = pt["xs_h{}_bbH".format(i + 1)]
        xs_tt = pt["xs_h{}_ttH".format(i + 1)]
        xs_vbf = pt["xs_h{}_VBF".format(i + 1)]
        xs_vh = pt["xs_h{}_VH".format(i + 1)]
        xs_tot = pt["xs_h{}_tot".format(i + 1)]

        neutral[i].setCxn("LHC13", "ggH", xs_gg)
        neutral[i].setCxn("LHC13", "bbH", xs_bb)
        neutral[i].setCxn("LHC13", "Htt", xs_tt)
        neutral[i].setCxn("LHC13", "vbfH", xs_vbf)
        # neutral[i].setCxn('LHC13','qqHZ',xs_zh)   # ???????????
        # neutral[i].setCxn('LHC13','HW',)     # are these going into the correct channel?
        # 		print('h{} prod'.format(i+1),xs_gg,xs_bb)
        missing_xs = (xs_tot - xs_gg - xs_bb - xs_tt - xs_vbf) / xs_tot * 100
        if flags and missing_xs > 1:
            print(" [pt {}] [h{}] [XS]".format(int(pt["index"]), i + 1))
            print("        xs_gg/xs_tot : {} %".format(round(xs_gg / xs_tot * 100, 3)))
            print("        xs_tt/xs_tot : {} %".format(round(xs_tt / xs_tot * 100, 3)))
            print("        xs_bb/xs_tot : {} %".format(round(xs_bb / xs_tot * 100, 3)))
            print(
                "        xs_vbf/xs_tot : {} %".format(round(xs_vbf / xs_tot * 100, 3))
            )
            # 			print("        xs_hv/xs_tot : {} %".format(round((xs_hw+xs_hz)/xs_tot*100,3)))
            print(
                "        (xs_tot-Soma)/xs_tot : {} %".format(
                    round(
                        (xs_tot - xs_gg - xs_tt - xs_bb - xs_vbf - xs_vh)
                        / xs_tot
                        * 100,
                        5,
                    )
                )
            )


def SetChargedScalarCrossSections(pt, flags):
    for i in range(Nc):
        etaR = pt["etaR_quark_Hp{}".format(i + 1)]
        etaL = pt["etaL_quark_Hp{}".format(i + 1)]
        br_tHpb = pt["BR_tHp{}b".format(i + 1)]
        xs_Hptb = HP.EffectiveCouplingCxns.ppHpmtb(
            "LHC13",
            mass=pt["mHp{}".format(i + 1)],
            cHpmtbR=etaR,
            cHpmtbL=etaL,
            brtHpb=br_tHpb,
        )
        charged[i].setCxn("LHC13", "Hpmtb", xs_Hptb)
        charged[i].setCxn("LHC13", "brtHpb", br_tHpb)  #### Needed for masses < 145 GeV
        # 		print('xs charged',i,xs_Hptb,etaR,etaL,br_tHpb)
        if flags and 0:
            print(" [pt {}] [Hp{}] [XS]".format(int(pt["index"]), i + 1))
            print("        etaR : {}".format(etaR))
            print("        etaL : {}".format(etaL))
            print("        br_tHpb : {} %".format(br_tHpb * 100))
            print("        xs_Hptb : {} ".format(xs_Hptb))


def SetNeutralScalarDecayWidths(pt, flags):
    for i in range(Nn):
        w = pt["Wt_h{}".format(i + 1)]
        neutral[i].setTotalWidth(0.)

        # 		print('width',i,w)
        # to gauge bosons
        for vv in ["WW", "ZZ", "gamgam", "Zgam", "gg"]:
            width = w * pt["BR_h{}{}".format(i + 1, vv)]
            neutral[i].setDecayWidth(vv, width)
        # 			print(vv,width)
        # to fermions
        for ff in ["ee", "mumu", "tautau", "dd", "uu", "ss", "cc", "bb", "tt"]:
            width = w * pt["BR_h{}{}".format(i + 1, ff)]
            neutral[i].setDecayWidth(ff, width)
        # 			print("        width h{} > {} : {}".format(i+1,ff,width))
        # 			if i==2 and  ff=='bb' and flags: print("        width h{} > bb : {}".format(i+1,width))

        #################################################################################################
        #               Below we have several tests on how to access variables
        #
        # 		print(neutral[0].cxn("LHC8", "ggH")*pt['BR_h{}{}'.format(1,'tautau')])
        # 		print(neutral[1].cxn("LHC8", "ggH")*pt['BR_h{}{}'.format(2,'tautau')])
        # 		print(neutral[2].cxn("LHC8", "ggH")*pt['BR_h{}{}'.format(3,'tautau')])
        # 		print(pt['BR_h{}{}'.format(3,'WW')])
        # 		print(neutral[2].br("WW"))
        # 		print(neutral[0].totalWidth())
        # 		print(pt["Wt_h{}".format(1)])
        # 		print(neutral[0].cxn("LHC8", "ggH"))
        # 		print(neutral[0].cxn("LHC13", "ggH"))
        ##################################################################################################
        # to hZ
        for j in range(Nn):
            width = w * pt["BR_h{}h{}Z".format(i + 1, j + 1)]
            neutral[i].setDecayWidth("h{}".format(j + 1), "Z", width)
        # 			print("h{}".format(i+1),"h{}".format(j+1),"Z",width)
        # to HpW
        for j in range(Nc):
            width = w * pt["BR_h{}Hp{}W".format(i + 1, j + 1)]
            neutral[i].setDecayWidth("Hp{}".format(j + 1), "W", width)
        # 			print("Hp{}".format(j+1),"W",width)
        # to hh
        for j in range(Nn):
            for k in range(j, Nn):
                if i != j and i != k:  # these are always 0
                    width = w * pt["BR_h{}h{}h{}".format(i + 1, j + 1, k + 1)]
                    neutral[i].setDecayWidth(
                        "h{}".format(j + 1), "h{}".format(k + 1), width
                    )
        # 					if flags: print("        width h{} > h{} h{} : {}".format(i+1,j+1,k+1,width))
        # 		if i==2: print('decay h{}'.format(i+1),neutral[i].totalWidth())
        # 		print(3,2,1,pt["BR_h{}h{}h{}".format(3,2,1)])
        # 		print(3,1,2,pt["BR_h{}h{}h{}".format(3,1,2)])
        if w > 0:
            missing = (w - neutral[i].totalWidth()) / w
        else:
            missing = 0
        # 		print(missing)
        if (missing < 0 or missing * 100 > 1e-3) and flags:
            print(" [pt {}] [h{}] [Widths]".format(int(pt["index"]), i + 1))
            print("        mass = {}".format(neutral[i].mass()))
            print("        w_total = {}".format(w))
            print("        missing = {}".format(missing))
            print("        missing/w_total = {} %".format(missing / w * 100))
            print(
                "        br_invisible = {} %".format(
                    pt["BR_h{}invisible".format(i + 1)] * 100
                )
            )


def SetChargedScalarDecayWidths(pt, flags):
    for i in range(Nc):
        w = pt["Wt_Hp{}".format(i + 1)]
        charged[i].setTotalWidth(0.)
        br_tot = 0

        # to WZ
        br = pt["BR_Hp{}WZ".format(i + 1)]
        br_tot += br
        charged[i].setDecayWidth("WZ", w * br)
        # to hW
        for j in range(Nn):
            br = pt["BR_Hp{}h{}W".format(i + 1, j + 1)]
            br_tot += br
            charged[i].setDecayWidth("W", "h{}".format(j + 1), w * br)
        # to Hph
#        print(charged[i].totalWidth())
        for j in range(Nc):
            for k in range(Nn):
                br = pt['BR_Hp{}Hp{}h{}'.format(i+1,j+1,k+1)]
                br_tot += br
                charged[i].setDecayWidth('Hp{}'.format(j+1),'h{}'.format(k+1),w*br)
        # to QQ and LL
        charged[i].setDecayWidth("ud", w * pt["BR_Hp{}QQ11".format(i + 1)])
        charged[i].setDecayWidth("us", w * pt["BR_Hp{}QQ12".format(i + 1)])
        charged[i].setDecayWidth("ub", w * pt["BR_Hp{}QQ13".format(i + 1)])
        charged[i].setDecayWidth("cd", w * pt["BR_Hp{}QQ21".format(i + 1)])
        charged[i].setDecayWidth("cs", w * pt["BR_Hp{}QQ22".format(i + 1)])
        charged[i].setDecayWidth("cb", w * pt["BR_Hp{}QQ23".format(i + 1)])
        charged[i].setDecayWidth("t", "d", w * pt["BR_Hp{}QQ31".format(i + 1)])
        charged[i].setDecayWidth("t", "s", w * pt["BR_Hp{}QQ32".format(i + 1)])
        charged[i].setDecayWidth("tb", w * pt["BR_Hp{}QQ33".format(i + 1)])
        charged[i].setDecayWidth("enu", w * pt["BR_Hp{}LL1".format(i + 1)])
        charged[i].setDecayWidth("munu", w * pt["BR_Hp{}LL2".format(i + 1)])
        charged[i].setDecayWidth("taunu", w * pt["BR_Hp{}LL3".format(i + 1)])

        for j in range(3):
            br = pt["BR_Hp{}LL{}".format(i + 1, j + 1)]
            br_tot += br
        for k in range(3):
            for j in range(3):
                br = pt["BR_Hp{}QQ{}{}".format(i + 1, k + 1, j + 1)]
                br_tot += br
        # 		print('decay x{}'.format(i+1),charged[i].totalWidth(),'decay')
        missing = w - charged[i].totalWidth()
        if abs(missing) < 1e-6:
            missing = 0  # ignore very small decays, error comes from file priting
        if (missing < 0 or (missing / w) * 100 > 1e-5) and flags:
            print(" [pt {}] [Hp{}] [Widths]".format(int(pt["index"]), i + 1))
            print("        Sum(br) = {}".format(br_tot))
            print("        mass = {}".format(charged[i].mass()))
            print("        w_total = {}".format(w))
            print("        chargedwidth= {}".format(charged[i].totalWidth()))
            print("        missing = {}".format(missing))
            print("        missing/w_total = {} %".format(missing / w * 100))


def SetNeutralScalarCouplings(pt, flags):
    for i in range(Nn):
        cpls = HP.NeutralEffectiveCouplings()
        cpls.uu = pt["g_h{}uu_s".format(i + 1)] + 1j * pt["g_h{}uu_p".format(i + 1)]
        cpls.cc = cpls.uu
        cpls.tt = cpls.uu
        cpls.dd = pt["g_h{}dd_s".format(i + 1)] + 1j * pt["g_h{}dd_p".format(i + 1)]
        cpls.ss = cpls.dd
        cpls.bb = cpls.dd
        cpls.ee = pt["g_h{}ee_s".format(i + 1)] + 1j * pt["g_h{}ee_p".format(i + 1)]
        cpls.mumu = cpls.ee
        cpls.tautau = cpls.ee
        cpls.ZZ = pt["g_h{}ZZ".format(i + 1)]
        cpls.WW = pt["g_h{}WW".format(i + 1)]
        cpls.gamgam = pt["g_h{}gaga".format(i + 1)]
        cpls.Zgam = pt["g_h{}Zga".format(i + 1)]
        cpls.gg = pt["g_h{}gg".format(i + 1)]
        # 		print('coup h{}'.format(i+1),cpls)

        if i == 0:
            HP.effectiveCouplingInput(
                neutral[i], cpls, reference=HP.ReferenceModel.SMHiggs
            )
        else:
            HP.effectiveCouplingInput(neutral[i], cpls)


def SetScalarsProperties(
    pt, neutralCS, chargedCS, Couplings, neutralW, chargedW, flags
):

    # MASSES
    if 0:  # confirm entries being read
        print(pt)
    SetScalarMasses(pt)

    # TOTAL WIDTHS
    # SetScalarTotalWidths(pt)  # not to be done when decay widths are set

    # COUPLINGS
    # neutrals - NeutralEffectiveCouplings
    if Couplings:
        SetNeutralScalarCouplings(pt, flags)
    # charged - do not receive couplings

    # PRODUCTION CROSS SECTIONS LHC13
    # neutrals gg,bbH,ttHBF,VH(ZH,WH)
    if neutralCS:
        SetNeutralScalarCrossSections(pt, flags)

    # DECAY WIDTHS
    # neutral scalars - vv,ff,hZ,HpW,hh
    if neutralW:
        SetNeutralScalarDecayWidths(pt, flags)

    # charged compute xs_Hptb from cHpmtbR,cHpmtbL,brtHpb
    if chargedCS:
        SetChargedScalarCrossSections(pt, flags)

    # charged hHp,hW,WZ,QQ,LL
    if chargedW:
        SetChargedScalarDecayWidths(pt, flags)


#####################################################################################################################################################
##################### To Run
#####################################################################################################################################################


def RunHiggstoolsPoint(pt, neutralCS, chargedCS, Couplings, neutralW, chargedW, flags):
    SetScalarsProperties(pt, neutralCS, chargedCS, Couplings, neutralW, chargedW, flags)
    return bounds(pred), signals(pred)


def ProcessData(df, neutralCS, chargedCS, Couplings, neutralW, chargedW, flags):

    # list of dicts with points
    df_list = df.to_dict("records")

    # prepare output dataframe
    # 	res = df[['index','mh1','mh2','mh3','mHp1']].copy()
    res = df[["mh1", "mh2", "mh3", "mh4", "mh5", "mHp1", "mHp2"]].copy()
    res = res.round(1)
    res["HBres"] = [-1 for _ in range(len(df))]
    for scalar in ["h1", "h2", "h3", "h4", "h5", "Hp1", "Hp2"]:
        res["obsRatio_{}".format(scalar)] = ["-" for _ in range(len(df))]
        res["key_{}".format(scalar)] = ["-" for _ in range(len(df))]
        res["description_{}".format(scalar)] = ["-" for _ in range(len(df))]

    res['chisq']= ["-" for _ in range(len(df))]
    res['chisqdiff']= ["-" for _ in range(len(df))]
                

    # run point by point and save results to res
    for i, pt in enumerate(df_list):

        reshb, chisq = RunHiggstoolsPoint(
            pt, neutralCS, chargedCS, Couplings, neutralW, chargedW, flags
        )
        brtop = pt["BR_tHp{}b".format(1)]
        # 		print(i,brtop)
        # 		print(i,chisq)
        # 		print(i,int(reshb.allowed),neutral[1].cxn("LHC13", "ggH")*pt['BR_h{}h{}h{}'.format(2,1,1)])
        # 		print(neutral[2].cxn("LHC13", "ggH")*pt['BR_h{}h{}h{}'.format(3,1,1)])
        # 		print(neutral[3].cxn("LHC13", "ggH")*pt['BR_h{}h{}h{}'.format(4,1,1)])
        res.at[i, "HBres"] = int(reshb.allowed)
        res.at[i,'chisq'] = int(chisq)
        chisqdiff=chisq-152.54
        res.at[i,'chisqdiff'] = float(chisqdiff)
        limits = reshb.selectedLimits
        key0 = "-"
        description = "-"
        for scalar in ["h1", "h2", "h3", "h4", "h5", "Hp1", "Hp2"]:
            if scalar in limits.keys():
                obs = limits[scalar].obsRatio()
                exp = limits[scalar].expRatio()
                limit = limits[scalar].limit()
                key0 = limit.citeKey()
                description = limit.processDesc()
                res.at[i, "obsRatio_" + scalar] = "{} ".format(obs)
                res.at[i, "key_" + scalar] = "{} ".format(key0)
        # 				res.at[i,'description_'+scalar] = "{} ".format(description)
    return res


def ProcessDataml(df, neutralCS, chargedCS, Couplings, neutralW, chargedW, flags, Do_Chisq):

    # list of dicts with points
    df_list = df.to_dict("records")

    # prepare output dataframe
    res = pd.DataFrame()
    res["GoodHB"] = [-1 for _ in range(len(df))]
    for scalar in ["h1", "h2", "h3", "h4", "h5", "Hp1", "Hp2"]:
        res["selLim_{}_obsRatio".format(scalar)] = [0. for _ in range(len(df))]

    res['chisq']= ["-" for _ in range(len(df))]
    res['chisqdiff']= ["-" for _ in range(len(df))]

    # run point by point and save results to res
    for i, pt in enumerate(df_list):

        reshb, chisq = RunHiggstoolsPoint(
            pt, neutralCS, chargedCS, Couplings, neutralW, chargedW, flags
        )
        res.at[i, "GoodHB"] = int(reshb.allowed)
        res.at[i,'chisq'] = int(chisq)
        if Do_Chisq:
            chisqdiff=chisq-152.54
        else:
            chisqdiff=0
        res.at[i,'chisqdiff'] = float(chisqdiff)
        limits = reshb.selectedLimits
        if "h1" not in limits.keys():
        	res.at[i, "GoodHB"]=0
        for scalar in ["h1", "h2", "h3", "h4", "h5", "Hp1", "Hp2"]:
            if scalar in limits.keys():
                obs = limits[scalar].obsRatio()
                res.at[i, "selLim_" + scalar + "_obsRatio"] = obs
                # res.at[i, "selLim_" + scalar + "_obsRatio"] = "{} ".format(obs)
                # res.at[i, "selLim_" + scalar + "_obsRatio"] = pd.to_numeric(
                #     res.at[i, "selLim_" + scalar + "_obsRatio"], errors="ignore"
                # )

    # [{'GoodHB': 0, 'selLim_H1_obsRatio': 1.0221854048112649, 'selLim_H2_obsRatio': 0.08476174846402877,
    # 'selLim_H3_obsRatio': 0.0005995611486311407, 'selLim_H4_obsRatio': 2.710811666082386,
    # 'selLim_H5_obsRatio': 0.0005995611486311407, 'selLim_H1+_obsRatio': 0.0,
    #'selLim_H2+_obsRatio': 0.016321966785289795},

    # 	print(res.dtypes)
    return res
