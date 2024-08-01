import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


def plot_line_cross():
    # # D2S
    # # Semi-positive + Positive
    # ## Cross-Area
    # ### Triplet
    # y_cross_triplet = [0., 0.2766798418972332, 2.1080368906455864, 5.309617918313571, 10.223978919631094, 14.888010540184455, 19.683794466403164, 23.122529644268774, 26.956521739130434, 29.446640316205535, 32.62187088274045, 35.63899868247694, 37.733860342556, 40.355731225296445, 42.635046113306984, 44.13702239789196, 45.12516469038208, 46.35046113306983, 47.404479578392625, 48.32674571805007, 49.1699604743083, 50.59288537549407, 51.370223978919626, 52.18708827404479, 52.83267457180501, 53.57048748353096, 54.25559947299078, 55.15151515151515, 55.718050065876156, 56.46903820816864, 57.206851119894594, 57.62845849802372, 58.06324110671937, 58.35309617918314, 58.68247694334651, 59.14361001317523, 59.81554677206851, 60.487483530961796, 61.04084321475626, 61.48880105401845, 61.976284584980235, 62.1870882740448, 62.62187088274045, 63.306982872200265, 63.754940711462446, 64.00527009222662, 64.47957839262187, 64.72990777338603, 65.13833992094862, 65.37549407114625, 65.6258234519104]
    # ### InfoNCE
    # y_cross_infonce = [0., 0.6193173913043478, 3.311452173913044, 7.810982608695653, 14.370691304347826, 20.77873043478261, 28.04623043478261, 32.81118260869565, 37.51293913043478, 41.91135652173913, 46.853256521739134, 50.5059652173913, 53.85533478260871, 57.17942608695653, 59.833643478260875, 61.70423478260871, 63.473713043478256, 65.15471739130435, 65.91306521739132, 66.91155652173914, 68.06171739130436, 69.40146521739132, 70.29884347826086, 71.10774782608696, 71.8787347826087, 72.62444347826089, 73.39543043478261, 73.91363478260871, 74.50767391304349, 74.83629130434782, 75.35449565217392, 75.7968652173913, 76.2392347826087, 76.54257391304347, 76.84591304347828, 77.09869565217392, 77.60426086956522, 77.88232173913043, 78.09718695652175, 78.10982608695652, 78.23621739130436, 78.4131652173913, 78.67858695652174, 78.96928695652174, 79.33582173913044, 79.48749130434783, 79.77819130434784, 79.85402608695652, 80.04361304347826, 80.22056086956523, 80.4607043478261]
    # ### InfoNCE w/. MES
    # y_cross_infonce_wmes = [0., 0.5138339920948617, 2.9907773386034253, 7.470355731225296, 14.729907773386033, 22.200263504611332, 29.683794466403164, 34.54545454545455, 39.644268774703555, 43.70223978919631, 48.40579710144928, 52.62187088274045, 56.42951251646904, 59.84189723320158, 62.740447957839265, 64.69038208168642, 65.99472990777339, 67.49670619235837, 68.51119894598156, 69.433465085639, 70.39525691699605, 71.51515151515152, 72.66139657444005, 73.21475625823452, 73.8076416337286, 74.66403162055336, 75.48089591567853, 76.10013175230567, 76.71936758893281, 77.19367588932806, 77.62845849802372, 77.99736495388669, 78.26086956521739, 78.53754940711462, 78.82740447957839, 79.10408432147563, 79.61791831357048, 79.90777338603425, 80.02635046113306, 80.15810276679842, 80.3425559947299, 80.36890645586297, 80.47430830039526, 80.71146245059289, 80.97496706192359, 81.10671936758894, 81.26482213438734, 81.40974967061923, 81.67325428194994, 81.85770750988142, 81.9235836627141]
    # ### weighted-InfoNCE w/. MES
    # y_cross_winfonce_wmes = [0., 0.7114624505928854, 3.438735177865613, 8.49802371541502, 15.942028985507244, 23.689064558629777, 32.1870882740448, 37.351778656126484, 43.3201581027668, 48.11594202898551, 53.84716732542819, 58.484848484848484, 62.450592885375485, 66.16600790513833, 69.45981554677206, 71.33069828722003, 72.7931488801054, 74.46640316205534, 75.41501976284584, 76.3372859025033, 77.25955204216073, 78.23451910408433, 79.15678524374177, 79.9604743083004, 80.54018445322794, 81.37022397891963, 81.81818181818183, 82.43741765480897, 82.99077733860342, 83.42555994729908, 83.82081686429513, 84.18972332015811, 84.33465085639, 84.54545454545455, 84.83530961791831, 85.2832674571805, 85.5467720685112, 85.79710144927536, 86.12648221343873, 86.23188405797102, 86.37681159420289, 86.50856389986825, 86.58761528326747, 86.93017127799737, 87.03557312252964, 87.19367588932806, 87.31225296442688, 87.39130434782608, 87.48353096179183, 87.53623188405797, 87.62845849802372]

    # S2D
    # Semi-positive + Positive
    ## Cross-Area
    ### Triplet
    y_cross_triplet = [0., 0.5524861878453038, 4.23572744014733, 9.760589318600369, 18.96869244935543, 25.78268876611418, 34.06998158379374, 39.963167587476974, 44.935543278084715, 48.25046040515654, 52.117863720073665, 53.77532228360957, 55.06445672191529, 57.6427255985267, 60.40515653775322, 62.61510128913444, 64.27255985267035, 65.19337016574586, 66.66666666666666, 67.40331491712708, 67.58747697974218, 68.13996316758748, 68.87661141804789, 69.42909760589319, 69.9815837937385, 70.71823204419888, 71.27071823204419, 72.55985267034991, 73.11233885819522, 73.66482504604052, 74.21731123388582, 74.40147329650092, 74.76979742173113, 74.76979742173113, 74.95395948434623, 75.50644567219152, 75.87476979742172, 76.24309392265194, 76.24309392265194, 76.61141804788214, 76.97974217311234, 77.16390423572744, 77.90055248618785, 78.26887661141805, 78.26887661141805, 78.26887661141805, 78.63720073664825, 78.63720073664825, 78.82136279926335, 79.00552486187846, 79.00552486187846]
    ### InfoNCE
    y_cross_infonce = [0., 0.9208103130755065, 5.156537753222836, 13.627992633517497, 25.41436464088398, 34.43830570902394, 45.30386740331492, 51.01289134438306, 57.0902394106814, 61.141804788213626, 64.82504604051566, 67.58747697974218, 69.61325966850829, 72.37569060773481, 74.21731123388582, 76.97974217311234, 78.26887661141805, 79.74217311233886, 80.66298342541437, 81.95211786372008, 82.13627992633518, 83.05709023941068, 83.97790055248619, 84.34622467771639, 84.71454880294658, 84.8987108655617, 85.6353591160221, 86.1878453038674, 86.55616942909761, 87.10865561694291, 87.47697974217311, 87.66114180478822, 87.66114180478822, 88.02946593001842, 88.02946593001842, 88.02946593001842, 88.39779005524862, 88.39779005524862, 88.39779005524862, 88.39779005524862, 88.58195211786372, 88.76611418047882, 89.13443830570903, 89.68692449355433, 89.68692449355433, 89.68692449355433, 89.87108655616943, 89.87108655616943, 89.87108655616943, 90.05524861878453, 90.60773480662984]
    ### InfoNCE w/. MES
    y_cross_infonce_wmes = [0., 1.841620626151013, 6.998158379373849, 16.206261510128915, 27.255985267034994, 37.75322283609576, 48.43462246777164, 54.32780847145487, 57.826887661141804, 62.246777163904234, 65.74585635359117, 69.42909760589319, 70.90239410681399, 74.58563535911603, 77.71639042357275, 80.47882136279927, 82.68876611418048, 83.79373848987109, 84.5303867403315, 85.0828729281768, 85.451197053407, 85.6353591160221, 86.0036832412523, 87.47697974217311, 87.84530386740332, 88.58195211786372, 88.76611418047882, 88.95027624309392, 89.13443830570903, 89.13443830570903, 89.13443830570903, 89.13443830570903, 89.13443830570903, 89.31860036832413, 89.31860036832413, 89.68692449355433, 89.68692449355433, 89.87108655616943, 90.05524861878453, 90.23941068139963, 90.23941068139963, 90.42357274401473, 90.60773480662984, 90.79189686924494, 91.16022099447514, 91.16022099447514, 91.52854511970534, 91.52854511970534, 91.52854511970534, 91.52854511970534, 91.71270718232044]
    ### weighted-InfoNCE w/. MES
    y_cross_winfonce_wmes = [0., 2.0257826887661143, 9.023941068139964, 19.88950276243094, 33.88581952117864, 47.51381215469613, 61.51012891344383, 67.77163904235728, 71.08655616942909, 73.84898710865562, 77.71639042357275, 79.37384898710866, 80.84714548802947, 83.79373848987109, 85.451197053407, 87.66114180478822, 88.95027624309392, 89.68692449355433, 90.42357274401473, 90.42357274401473, 90.42357274401473, 90.42357274401473, 90.79189686924494, 90.97605893186004, 91.16022099447514, 91.16022099447514, 91.52854511970534, 91.52854511970534, 91.52854511970534, 91.52854511970534, 91.89686924493554, 92.08103130755065, 92.26519337016575, 92.26519337016575, 92.44935543278085, 92.44935543278085, 92.44935543278085, 92.44935543278085, 92.63351749539595, 92.63351749539595, 92.81767955801105, 92.81767955801105, 92.81767955801105, 93.18600368324125, 93.18600368324125, 93.18600368324125, 93.37016574585635, 93.37016574585635, 93.37016574585635, 93.55432780847146, 93.55432780847146]

    dis_threshold_cross_list = [10 * (i) for i in range(51)]
    dis_threshold_same_list = [10 * (i) for i in range(21)]

    plt.figure(figsize=(10, 6), dpi=300)

    y = np.array(y_cross_triplet)
    x = np.array(dis_threshold_cross_list)
    x_new = np.linspace(x.min(), x.max(), 500)
    spl = make_interp_spline(x, y, k=3)  
    y_smooth = spl(x_new)
    plt.plot(x_new, y_smooth, label='Triplet', color='green')

    y = np.array(y_cross_infonce)
    x = np.array(dis_threshold_cross_list)
    x_new = np.linspace(x.min(), x.max(), 500)
    spl = make_interp_spline(x, y, k=3)  
    y_smooth = spl(x_new)
    plt.plot(x_new, y_smooth, label='InfoNCE', color='orange')

    y = np.array(y_cross_infonce_wmes)
    x = np.array(dis_threshold_cross_list)
    x_new = np.linspace(x.min(), x.max(), 500)
    spl = make_interp_spline(x, y, k=3)  
    y_smooth = spl(x_new)
    plt.plot(x_new, y_smooth, label='InfoNCE w/. MES', color='purple')

    y = np.array(y_cross_winfonce_wmes)
    x = np.array(dis_threshold_cross_list)
    x_new = np.linspace(x.min(), x.max(), 500)
    spl = make_interp_spline(x, y, k=3)  
    y_smooth = spl(x_new)
    plt.plot(x_new, y_smooth, label='Weighted-InfoNCE w/. MES', color='red')

    plt.xlabel('Threshold (m)', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.xlim(0, 500)  # Set x-axis to start from 0
    plt.ylim(0, None)  # Set y-axis to start from 0

    # 调整边框
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)

    # 显示图表
    plt.tight_layout()
    plt.savefig('/home/xmuairmud/jyx/GTA-UAV/Sample4Geo/images/plot_acc_threshold_cross_area_s2d.png', bbox_inches='tight')
    # print(y.tolist())


def plot_line_same():
    # # D2S
    # # Semi-positive + Positive
    # ## Same-Area
    # ### Triplet
    # y_same_triplet = [0., 0.894251242015614, 5.365507452093683, 13.637331440738114, 24.080908445706175, 35.32292405961675, 50.62100780695529, 59.30801987224983, 66.27040454222853, 71.44428672817601, 75.7877927608233, 79.77998580553584, 82.71823988644428, 84.47480482611782, 85.52874378992193, 86.23136976579134, 86.87012065294536, 87.18949609652235, 87.41305890702627, 87.50887154009936, 87.73243435060327]
    # ### InfoNCE
    # y_same_infonce = [0., 0.7097232079488999, 4.435770049680625, 12.029808374733854, 21.788502484031227, 32.50532292405961, 46.41589779985806, 54.32931156848829, 61.320085166784956, 67.67210787792762, 73.4918381831086, 78.49538679914833, 83.32150461320084, 86.8346344925479, 89.53158268275374, 91.5542938254081, 93.22214336408801, 94.3222143364088, 95.24485450674237, 95.6706884315117, 96.20298083747339]
    # ### InfoNCE w/. MES
    # y_same_infonce_wmes =  [0., 0.8516678495386799, 4.258339247693399, 11.391057487579843, 21.50461320085167, 32.00851667849538, 45.88360539389638, 53.76153300212917, 61.213626685592615, 67.81405251951739, 73.98864442867283, 80.23420865862315, 86.16039744499645, 89.7444996451384, 92.72533711852378, 94.99645138396026, 96.27395315826828, 97.30305180979418, 97.94180269694819, 98.22569198012775, 98.54506742370475]
    # ### weighted-InfoNCE w/. MES
    # y_same_winfonce_wmes = [0., 0.8161816891412349, 4.151880766501065, 10.716820440028389, 20.298083747338538, 30.837473385379706, 44.357700496806245, 52.4485450674237, 59.90063875088716, 66.82044002838893, 73.13697657913414, 79.45351312987935, 85.94748048261178, 89.63804116394606, 93.22214336408801, 96.02555003548616, 97.76437189496096, 98.65152590489708, 99.11284599006387, 99.32576295244854, 99.46770759403833]

    # S2D
    # Semi-positive + Positive
    ## Same-Area
    ### Triplet
    y_same_triplet = [0., 1.7948717948717947, 8.461538461538462, 21.153846153846153, 38.205128205128204, 53.58974358974359, 70.64102564102565, 80.51282051282051, 86.7948717948718, 91.15384615384615, 92.56410256410257, 94.74358974358974, 96.15384615384616, 96.92307692307692, 97.3076923076923, 97.82051282051282, 98.46153846153847, 99.23076923076923, 99.35897435897436, 99.48717948717949, 99.74358974358975, 99.74358974358975, 99.74358974358975, 99.74358974358975, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0][:21]
    ### InfoNCE
    y_same_infonce = [0., 1.6666666666666667, 8.076923076923077, 20.76923076923077, 35.76923076923077, 52.56410256410257, 70.25641025641025, 80.12820512820514, 86.02564102564104, 90.76923076923077, 93.2051282051282, 95.25641025641025, 96.02564102564102, 96.66666666666667, 97.17948717948718, 97.6923076923077, 98.2051282051282, 99.1025641025641, 99.23076923076923, 99.23076923076923, 99.48717948717949, 99.61538461538461, 99.61538461538461, 99.61538461538461, 99.61538461538461, 99.61538461538461, 99.87179487179488, 99.87179487179488, 99.87179487179488, 99.87179487179488, 99.87179487179488, 99.87179487179488, 99.87179487179488, 99.87179487179488, 99.87179487179488, 99.87179487179488, 99.87179487179488, 99.87179487179488, 99.87179487179488, 99.87179487179488, 99.87179487179488, 99.87179487179488, 99.87179487179488, 99.87179487179488, 99.87179487179488, 99.87179487179488, 99.87179487179488, 99.87179487179488, 99.87179487179488, 99.87179487179488, 99.87179487179488][0:21]
    ### InfoNCE w/. MES
    y_same_infonce_wmes =  [0., 1.6666666666666667, 7.948717948717948, 21.28205128205128, 36.794871794871796, 54.61538461538461, 71.66666666666667, 79.87179487179488, 85.76923076923076, 89.23076923076924, 91.66666666666666, 94.35897435897435, 94.74358974358974, 95.38461538461539, 96.28205128205128, 96.92307692307692, 97.94871794871794, 98.97435897435898, 99.1025641025641, 99.23076923076923, 99.35897435897436, 99.35897435897436, 99.61538461538461, 99.74358974358975, 99.87179487179488, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0][:21]
    ### weighted-InfoNCE w/. MES
    y_same_winfonce_wmes = [0., 1.6666666666666667, 8.461538461538462, 22.435897435897438, 38.71794871794872, 55.64102564102564, 73.33333333333333, 82.17948717948718, 88.97435897435896, 92.82051282051282, 94.48717948717949, 95.8974358974359, 96.66666666666667, 97.17948717948718, 97.82051282051282, 98.2051282051282, 98.84615384615385, 99.61538461538461, 99.74358974358975, 99.74358974358975, 99.87179487179488, 99.87179487179488, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0][:21]

    dis_threshold_cross_list = [10 * (i) for i in range(51)]
    dis_threshold_same_list = [10 * (i) for i in range(21)]

    plt.figure(figsize=(10, 6), dpi=300)

    y = np.array(y_same_triplet)
    x = np.array(dis_threshold_same_list)
    x_new = np.linspace(x.min(), x.max(), 500)
    spl = make_interp_spline(x, y, k=3)  
    y_smooth = spl(x_new)
    plt.plot(x_new, y_smooth, label='Triplet', color='green')

    y = np.array(y_same_infonce)
    x = np.array(dis_threshold_same_list)
    x_new = np.linspace(x.min(), x.max(), 500)
    spl = make_interp_spline(x, y, k=3)  
    y_smooth = spl(x_new)
    plt.plot(x_new, y_smooth, label='InfoNCE', color='orange')

    y = np.array(y_same_infonce_wmes)
    x = np.array(dis_threshold_same_list)
    x_new = np.linspace(x.min(), x.max(), 500)
    spl = make_interp_spline(x, y, k=3)  
    y_smooth = spl(x_new)
    plt.plot(x_new, y_smooth, label='InfoNCE w/. MES', color='purple')

    y = np.array(y_same_winfonce_wmes)
    x = np.array(dis_threshold_same_list)
    x_new = np.linspace(x.min(), x.max(), 500)
    spl = make_interp_spline(x, y, k=3)  
    y_smooth = spl(x_new)
    plt.plot(x_new, y_smooth, label='Weighted-InfoNCE w/. MES', color='red')

    plt.xlabel('Threshold (m)', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.xlim(0, 200)  # Set x-axis to start from 0
    plt.ylim(0, None)  # Set y-axis to start from 0

    # 调整边框
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)

    # 显示图表
    plt.tight_layout()
    plt.savefig('/home/xmuairmud/jyx/GTA-UAV/Sample4Geo/images/plot_acc_threshold_same_area_s2d.png', bbox_inches='tight')
    # print(y.tolist())


if __name__ == '__main__':
    plot_line_same()
    # plot_line_cross()