#include <stan/math/rev.hpp>
#include <boost/math/differentiation/finite_difference.hpp>
#include <boost/math/special_functions/digamma.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>

namespace inv_gamma_test_internal {
struct TestValue {
  double y;
  double alpha;
  double beta;
  double value;
  double grad_y;
  double grad_alpha;
  double grad_beta;

  TestValue(double _y, double _alpha, double _beta, double _value,
            double _grad_y, double _grad_alpha, double _grad_beta)
      : y(_y),
        alpha(_alpha),
        beta(_beta),
        value(_value),
        grad_y(_grad_y),
        grad_alpha(_grad_alpha),
        grad_beta(_grad_beta) {}
};

// Test data generated in Mathematica (Wolfram Cloud). The code can be re-ran
// at https://www.wolframcloud.com/obj/martin.modrak/Published/InvGamma_test.nb
// but is also presented below for convenience:
//
// ig[y_,alpha_,beta_]:= -LogGamma[alpha] + alpha *Log[beta] - (alpha + 1) *
//    Log[y] - beta / y;
// igdy[y_,alpha_,beta_]= D[ig[y,alpha, beta],y];
// igdalpha[y_,alpha_,beta_]= D[ig[y,alpha, beta],alpha];
// igdbeta[y_,alpha_,beta_]= D[ig[y,alpha, beta],beta];
// out = OpenWrite["ig_test.txt"]
// alphas= {76*10^-7,811*10^-3,21*10^-1,2,621,  856,34251};
// betas=  {25*10^-7,31*10^-3,234*10^-2,368, 94256};
// ys = {10^-7,10^-3,3,961,64587};
//  WriteString[out, "std::vector<TestValue> testValues = {"];
//    Block[{$MaxPrecision = 80, $MinPrecision = 40}, {
//      For[i = 1, i <= Length[alphas], i++, {
//        For[j = 1, j <= Length[betas], j++, {
//        For[k = 1, k <= Length[ys], k++, {
//          calpha = alphas[[i]];
//          cbeta = betas[[j]];
//     cy=ys[[k]];
//          val = N[ig[cy,calpha,cbeta]];
//     dy = N[igdy[cy,calpha,cbeta]];
//    dalpha = N[igdalpha[cy,calpha,cbeta]];
//     dbeta = N[igdbeta[cy,calpha,cbeta]];
//          WriteString[out,"  TestValue(",CForm[cy],",",CForm[calpha],",
//            ",CForm[cbeta],",",
//            CForm[val],","CForm[dy],","CForm[dalpha],",",CForm[dbeta],"),"]
//        }]
//      }]
//   }]
//    }];
//  WriteString[out,"};"];
//  Close[out];
//  FilePrint[%]
std::vector<TestValue> testValues = {
    TestValue(1.e-7, 7.6e-6, 2.5e-6, -20.66923780946585, 2.39999924e8,
              131582.7434474094, -9.99999696e6),
    TestValue(0.001, 7.6e-6, 2.5e-6, -4.882148180028864, -997.5076,
              131573.5331070374, -996.96),
    TestValue(3, 7.6e-6, 2.5e-6, -12.886077429405958, -0.33333558888888887,
              131565.52673946976, 2.7066666666666666),
    TestValue(961, 7.6e-6, 2.5e-6, -18.655482566128377, -0.0010405906320484321,
              131559.75737734945, 3.038959417273673),
    TestValue(64587, 7.6e-6, 2.5e-6, -22.86330856495934,
              -0.00001548310960350059, 131555.5495833273, 3.0399845170080666),
    TestValue(1.e-7, 7.6e-6, 0.031, -309995.669166176, 3.099989999924e12,
              131592.16889916096, -9.999999999754839e6),
    TestValue(0.001, 7.6e-6, 0.031, -35.87957654659555, 29999.9924,
              131582.958558789, -999.9997548387097),
    TestValue(3, 7.6e-6, 0.031, -12.896338295972646, -0.32989142222222223,
              131574.95219122135, -0.3330881720430108),
    TestValue(961, 7.6e-6, 0.031, -18.655443188158124, -0.0010405570675707427,
              131569.18282910105, -0.0007954214360041623),
    TestValue(64587, 7.6e-6, 0.031, -22.86323741146007, -0.00001548310217268568,
              131564.97503507888, 0.00022967829838921945),
    TestValue(1.e-7, 7.6e-6, 2.34, -2.3399995669133317e7, 2.33999989999924e14,
              131596.49281816484, -9.999999999996752e6),
    TestValue(0.001, 7.6e-6, 2.34, -2344.879543684811, 2.3389999924e6,
              131587.28247779285, -999.9999967521368),
    TestValue(3, 7.6e-6, 2.34, -13.665972100854882, -0.07333586666666667,
              131579.2761102252, -0.33333008547008547),
    TestValue(961, 7.6e-6, 2.34, -18.657813031888782, -0.0010380568537152917,
              131573.5067481049, -0.0010373348630788796),
    TestValue(64587, 7.6e-6, 2.34, -22.863240299904014,
              -0.000015482548652188147, 131569.29895408274,
              -0.000012235128685497954),
    TestValue(1.e-7, 7.6e-6, 368, -3.679999995669095e9, 3.679999998999992e16,
              131601.55075017363, -9.999999999999978e6),
    TestValue(0.001, 7.6e-6, 368, -368004.8795052445, 3.679989999924e8,
              131592.34040980166, -999.9999999793479),
    TestValue(3, 7.6e-6, 368, -135.55260032723828, 40.55555302222222,
              131584.334042234, -0.3333333126811594),
    TestValue(961, 7.6e-6, 368, -19.038274071314152, -0.0006421156677541713,
              131578.56468011372, -0.00104056207415283),
    TestValue(64587, 7.6e-6, 368, -22.868863370451102, -0.000015394891525671157,
              131574.35688609155, -0.000015462339759448158),
    TestValue(1.e-7, 7.6e-6, 94256, -9.425599999956691e11, 9.42559999999e18,
              131607.0964369992, -1.e7),
    TestValue(0.001, 7.6e-6, 94256, -9.425600487946309e7, 9.42559989999924e10,
              131597.88609662725, -999.9999999999194),
    TestValue(3, 7.6e-6, 94256, -31431.552558180018, 10472.555553022223,
              131589.8797290596, -0.33333333325270187),
    TestValue(961, 7.6e-6, 94256, -116.73646293345952, 0.10102097591327105,
              131584.1103669393, -0.0010405826456952713),
    TestValue(64587, 7.6e-6, 94256, -24.322488369870648, 7.112225179539126e-6,
              131579.90257291714, -0.000015482911301889465),
    TestValue(1.e-7, 0.811, 2.5e-6, -6.412978783955374, 2.3189e8,
              4.158854833244236, -9.6756e6),
    TestValue(0.001, 0.811, 2.5e-6, 1.9045948023957546, -1808.5,
              -5.0514855387319475, 323400.),
    TestValue(3, 0.811, 2.5e-6, -12.592437695952176, -0.6036663888888889,
              -13.057853106382193, 324399.6666666667),
    TestValue(961, 0.811, 2.5e-6, -23.04075166508755, -0.0018844953146707004,
              -18.827215226684377, 324399.9989594173),
    TestValue(64587, 0.811, 2.5e-6, -30.661066636655015,
              -0.00002803969839071783, -23.035009248843522, 324399.999984517),
    TestValue(1.e-7, 0.811, 0.031, -309973.7689374134, 3.09998189e12,
              13.584306584837364, -9.999973838709677e6),
    TestValue(0.001, 0.811, 0.031, -21.448863827062215, 29189.,
              4.3739662128611805, -973.8387096774194),
    TestValue(3, 0.811, 0.031, -4.958728825410149, -0.6002222222222222,
              -3.632401354789066, 25.827956989247312),
    TestValue(961, 0.811, 0.031, -15.396742550008582, -0.0018844617501930113,
              -9.401763475091249, 26.16024973985432),
    TestValue(64587, 0.811, 0.031, -23.01702574604703, -0.00002803969095990292,
              -13.609557497250394, 26.16127483958871),
    TestValue(1.e-7, 0.811, 2.34, -2.3399970262239102e7, 2.3399998189e14,
              17.908225588703967, -9.999999653418804e6),
    TestValue(0.001, 0.811, 2.34, -2326.9421655149263, 2.338189e6,
              8.697885216727782, -999.6534188034188),
    TestValue(3, 0.811, 2.34, -2.221697179941002, -0.3436666666666667,
              0.6915176490775349, 0.013247863247863248),
    TestValue(961, 0.811, 2.34, -11.892446943387858, -0.0018819615363375602,
              -5.0778444712246475, 0.34554061385486984),
    TestValue(64587, 0.811, 2.34, -19.51036318413959, -0.00002803913743940539,
              -9.285638493383793, 0.34656571358926325),
    TestValue(1.e-7, 0.811, 368, -3.6799999661602564e9, 3.679999998189e16,
              22.966157597503287, -9.999999997796196e6),
    TestValue(0.001, 0.811, 368, -367982.8401826558, 3.67998189e8,
              13.755817225527103, -999.9977961956522),
    TestValue(3, 0.811, 368, -120.00638098747142, 40.285222222222224,
              5.7494496578768555, -0.33112952898550724),
    TestValue(961, 0.811, 368, -8.170963563960244, -0.0014860203503764397,
              -0.019912462425326627, 0.001163221621499344),
    TestValue(64587, 0.811, 368, -15.414041835833693, -0.0000279514803128884,
              -4.227706484584472, 0.002188321355892726),
    TestValue(1.e-7, 0.811, 94256, -9.425599999616626e11, 9.42559999998189e18,
              28.511844423095823, -9.999999999991396e6),
    TestValue(0.001, 0.811, 94256, -9.425597834263064e7, 9.4255998189e10,
              19.30150405111964, -999.9999913957732),
    TestValue(3, 0.811, 94256, -31411.508828971917, 10472.285222222223,
              11.295136483469394, -0.3333247291065467),
    TestValue(961, 0.811, 94256, -101.37164255776995, 0.10017707123064878,
              5.52577436316721, -0.0010319784995401194),
    TestValue(64587, 0.811, 94256, -12.370156966917563, -5.444363607678116e-6,
              1.3179803410080644, -6.878765146737539e-6),
    TestValue(1.e-7, 2.1, 2.5e-6, -2.16770285536294, 2.19e8, 2.7335398561883686,
              -9.16e6),
    TestValue(0.001, 2.1, 2.5e-6, -5.7222580084891135, -3097.5,
              -6.476800515787815, 839000.),
    TestValue(3, 2.1, 2.5e-6, -30.53949830153821, -1.0333330555555555,
              -14.483168083438061, 839999.6666666666),
    TestValue(961, 2.1, 2.5e-6, -48.4245200437431, -0.003225806448905872,
              -20.252530203740243, 839999.9989594172),
    TestValue(64587, 2.1, 2.5e-6, -61.468681509873704, -0.00004799727499282042,
              -24.460324225899388, 839999.999984517),
    TestValue(1.e-7, 2.1, 0.031, -309957.374254177, 3.099969e12,
              12.158991607781497, -9.999932258064516e6),
    TestValue(0.001, 2.1, 0.031, -16.926309330143543, 27900.,
              2.9486512358053134, -932.258064516129),
    TestValue(3, 2.1, 0.031, -10.75638212319264, -1.0298888888888889,
              -5.057716331844933, 67.40860215053763),
    TestValue(961, 2.1, 0.031, -28.631103620860593, -0.003225772884428183,
              -10.827078452147116, 67.74089490114464),
    TestValue(64587, 2.1, 0.031, -41.67523331146217, -0.00004799726756200551,
              -15.034872474306262, 67.74192000087903),
    TestValue(1.e-7, 2.1, 2.34, -2.339994829402427e7, 2.33999969e14,
              16.4829106116481, -9.999999102564102e6),
    TestValue(0.001, 2.1, 2.34, -2316.8460794220236, 2.3369e6,
              7.272570239671914, -999.1025641025641),
    TestValue(3, 2.1, 2.34, -2.4458188817394446, -0.7733333333333333,
              -0.7337973279783321, 0.5641025641025641),
    TestValue(961, 2.1, 2.34, -19.553276418255816, -0.003223272670572732,
              -6.503159448280515, 0.8963953147095707),
    TestValue(64587, 2.1, 2.34, -32.59503915357069, -0.00004799671404150798,
              -10.71095347043966, 0.897420414443964),
    TestValue(1.e-7, 2.1, 368, -3.679999937672367e9, 3.6799999969e16,
              21.54084262044742, -9.999999994293477e6),
    TestValue(0.001, 2.1, 368, -367966.2244222035, 3.679969e8,
              12.330502248471236, -999.9942934782608),
    TestValue(3, 2.1, 368, -113.71082832992754, 39.855555555555554,
              4.324134680820989, -0.32762681159420287),
    TestValue(961, 2.1, 368, -9.312118679485879, -0.0028273314846116115,
              -1.4452274394811937, 0.004665939012803692),
    TestValue(64587, 2.1, 368, -21.979043445922464, -0.00004790905691499099,
              -5.653021461640339, 0.005691038747197074),
    TestValue(1.e-7, 2.1, 94256, -9.425599999260265e11, 9.425599999969e18,
              27.086529446039957, -9.99999999997772e6),
    TestValue(0.001, 2.1, 94256, -9.425595457847987e7, 9.42559969e10,
              17.876189074063774, -999.9999777202512),
    TestValue(3, 2.1, 94256, -31398.064885996184, 10471.855555555556,
              9.869821506413526, -0.33331105358456403),
    TestValue(961, 2.1, 94256, -95.36440735510678, 0.09883576009641362,
              4.100459386111343, -0.0010183029775574339),
    TestValue(64587, 2.1, 94256, -11.786768258817553, -0.000025401940209780707,
              -0.10733463604780258, 6.796756835947913e-6),
    TestValue(1.e-7, 2, 2.5e-6, -2.4441526993052776, 2.2e8, 2.7960914897697338,
              -9.2e6),
    TestValue(0.001, 2, 2.5e-6, -5.07767381523383, -2997.5, -6.41424888220645,
              799000.),
    TestValue(3, 2, 2.5e-6, -29.094277351517903, -0.9999997222222222,
              -14.420616449856695, 799999.6666666666),
    TestValue(961, 2, 2.5e-6, -46.402362881692575, -0.003121748176273198,
              -20.18997857015888, 799999.9989594172),
    TestValue(64587, 2, 2.5e-6, -59.02574494560726, -0.0000464489757994843,
              -24.397772592318024, 799999.999984517),
    TestValue(1.e-7, 2, 0.031, -309958.5932491961, 3.09997e12,
              12.221543241362863, -9.999935483870968e6),
    TestValue(0.001, 2, 0.031, -17.224270312047572, 28000., 3.0112028693866786,
              -935.483870967742),
    TestValue(3, 2, 0.031, -10.253706348331644, -0.9965555555555555,
              -4.995164698263568, 64.18279569892474),
    TestValue(961, 2, 0.031, -27.551491633969377, -0.0031217146117955086,
              -10.76452681856575, 64.51508844953173),
    TestValue(64587, 2, 0.031, -40.17484192235504, -0.00004644896836866939,
              -14.972320840724896, 64.51611354926614),
    TestValue(1.e-7, 2, 2.34, -2.3399949945411187e7, 2.3399997e14,
              16.545462245229466, -9.999999145299146e6),
    TestValue(0.001, 2, 2.34, -2317.5764323043145, 2.337e6, 7.3351218732532795,
              -999.1452991452992),
    TestValue(3, 2, 2.34, -2.375535007265109, -0.74, -0.6712456943969669,
              0.5213675213675214),
    TestValue(961, 2, 2.34, -18.906056331751262, -0.0031192143979400576,
              -6.44060781469915, 0.853660271974528),
    TestValue(64587, 2, 2.34, -31.527039664850218, -0.00004644841484817186,
              -10.648401836858294, 0.8546853717089213),
    TestValue(1.e-7, 2, 368, -3.679999939829547e9, 3.679999997e16,
              21.603394254028785, -9.999999994565217e6),
    TestValue(0.001, 2, 368, -367967.4605682867, 3.67997e8, 12.393053882052602,
              -999.9945652173913),
    TestValue(3, 2, 368, -114.14633765633315, 39.888888888888886,
              4.386686314402354, -0.3278985507246377),
    TestValue(961, 2, 368, -9.170691793861259, -0.002723273211978937,
              -1.3826758058998285, 0.004394199882368909),
    TestValue(64587, 2, 368, -21.416837158081925, -0.00004636075772165487,
              -5.590469828058974, 0.005419299616762291),
    TestValue(1.e-7, 2, 94256, -9.425599999287382e11, 9.42559999997e18,
              27.149081079621322, -9.99999999997878e6),
    TestValue(0.001, 2, 94256, -9.425595636919463e7, 9.4255997e10,
              17.938740707645138, -999.9999787811917),
    TestValue(3, 2, 94256, -31399.05496400515, 10471.888888888889,
              9.932373139994892, -0.3333121145249816),
    TestValue(961, 2, 94256, -95.77754915204142, 0.09893981836904629,
              4.163011019692708, -0.00101936391797502),
    TestValue(64587, 2, 94256, -11.77913065353627, -0.000023853641016444586,
              -0.044783002466437405, 5.735816418361765e-6),
    TestValue(1.e-7, 621, 2.5e-6, -1380.5200261709679, -5.97e9,
              -3.2116498879958275, 2.384e8),
    TestValue(0.001, 621, 2.5e-6, -7084.354237540155, -621997.5,
              -12.421990259972011, 2.48399e8),
    TestValue(3, 621, 2.5e-6, -12064.312365451942, -207.33333305555556,
              -20.428357827622257, 2.4839999966666666e8),
    TestValue(961, 621, 2.5e-6, -15652.855603449167, -0.6472424557725271,
              -26.19771994792444, 2.4839999999895942e8),
    TestValue(64587, 621, 2.5e-6, -18270.103485229592, -0.009630420982550068,
              -30.405513970083586, 2.483999999999845e8),
    TestValue(1.e-7, 621, 0.031, -305502.3144884316, 3.09378e12,
              6.2138018635973005, -9.979967741935484e6),
    TestValue(0.001, 621, 0.031, -1262.1461998008217, -591000.,
              -2.9965385083788827, 19032.25806451613),
    TestValue(3, 621, 0.031, -6211.117160212609, -207.3298888888889,
              -11.002906076029129, 20031.924731182797),
    TestValue(961, 621, 0.031, -9799.650097965297, -0.6472424222080494,
              -16.77226819633131, 20032.2570239334),
    TestValue(64587, 621, 0.031, -12416.897947970194, -0.009630420975119253,
              -20.98006221849046, 20032.25804903314),
    TestValue(1.e-7, 621, 2.34, -2.339281716078703e7, 2.3399378e14,
              10.537720867463904, -9.999734615384616e6),
    TestValue(0.001, 621, 2.34, -885.9924983996625, 1.718e6, 1.3273804954877182,
              -734.6153846153846),
    TestValue(3, 621, 2.34, -3526.7331254781157, -207.07333333333332,
              -6.678987072162529, 265.05128205128204),
    TestValue(961, 621, 2.34, -7114.498799269653, -0.647239921994194,
              -12.448349192464711, 265.38357480188904),
    TestValue(64587, 621, 2.34, -9731.744282319263, -0.009630420421598757,
              -16.656143214623857, 265.38459990162346),
    TestValue(1.e-7, 621, 368, -3.6799896761850095e9, 3.679999378e16,
              15.595652876263223, -9.9999983125e6),
    TestValue(0.001, 621, 368, -363405.0167209353, 3.67378e8, 6.38531250428704,
              -998.3125),
    TestValue(3, 621, 368, -507.6440146804042, -166.44444444444446,
              -1.6210550633632073, 1.3541666666666667),
    TestValue(961, 621, 368, -3973.9035212849835, -0.6468439808082328,
              -7.39041718366539, 1.6864594172736733),
    TestValue(64587, 621, 368, -6590.774166365715, -0.009630332764472239,
              -11.598211205824535, 1.6874845170080666),
    TestValue(1.e-7, 621, 94256, -9.425599862323135e11, 9.42559999378e18,
              21.14133970185576, -9.99999999341156e6),
    TestValue(0.001, 621, 94256, -9.424796114520223e7, 9.4255378e10,
              11.930999329879576, -999.9934115600067),
    TestValue(3, 621, 94256, -28359.772495987443, 10265.555555555555,
              3.92463176222933, -0.32674489334012335),
    TestValue(961, 621, 94256, -627.7302336013836, -0.5451808892272076,
              -1.8447303580728533, 0.005547857266883238),
    TestValue(64587, 621, 94256, -3148.35631481939, -0.00960782564776703,
              -6.052524380231999, 0.00657295700127662),
    TestValue(1.e-7, 856, 2.5e-6, -2175.0105409363496, -8.32e9,
              -3.5328103253950243, 3.324e8),
    TestValue(0.001, 856, 2.5e-6, -10043.274739719935, -856997.5,
              -12.743150697371208, 3.42399e8),
    TestValue(3, 856, 2.5e-6, -16904.729246029532, -285.66666638888887,
              -20.749518265021454, 3.423999996666667e8),
    TestValue(961, 856, 2.5e-6, -21849.07258229777, -0.8917793964593117,
              -26.518880385323637, 3.423999999989594e8),
    TestValue(64587, 856, 2.5e-6, -25455.152059285596, -0.013268924086889951,
              -30.726674407482783, 3.423999999999845e8),
    TestValue(1.e-7, 856, 0.031, -304081.8238415726, 3.09143e12,
              5.892641426198104, -9.972387096774194e6),
    TestValue(0.001, 856, 0.031, -2006.0855403562186, -826000.,
              -3.3176989457780794, 26612.90322580645),
    TestValue(3, 856, 0.031, -8836.552879165813, -285.6632222222222,
              -11.324066513428326, 27612.56989247312),
    TestValue(961, 856, 0.031, -13780.885915189514, -0.8917793628948341,
              -17.09342863373051, 27612.902185223724),
    TestValue(64587, 856, 0.031, -17386.96536040181, -0.013268924079459137,
              -21.301222655889653, 27612.90321032346),
    TestValue(1.e-7, 856, 2.34, -2.3390380549174264e7, 2.3399143e14,
              10.216560430064707, -9.999634188034188e6),
    TestValue(0.001, 856, 2.34, -613.8108730464082, 1.483e6, 1.0062200580885214,
              -634.1880341880342),
    TestValue(3, 856, 2.34, -5136.04787852267, -285.4066666666667,
              -7.000147509561725, 365.4786324786325),
    TestValue(961, 856, 2.34, -10079.61365058522, -0.8917768626809786,
              -12.769509629863908, 365.8109252292395),
    TestValue(64587, 856, 2.34, -13685.690728842228, -0.013268923525938638,
              -16.977303652023053, 365.81195032897386),
    TestValue(1.e-7, 856, 368, -3.679986050959375e9, 3.679999143e16,
              15.274492438864026, -9.999997673913043e6),
    TestValue(0.001, 856, 368, -361944.2210735142, 3.67143e8, 6.064152066887843,
              -997.6739130434783),
    TestValue(3, 856, 368, -928.3447456571173, -244.77777777777777,
              -1.942215500762404, 1.9927536231884058),
    TestValue(961, 856, 368, -5750.404350532708, -0.8913809214950175,
              -7.7115776210645866, 2.3250463737954123),
    TestValue(64587, 856, 368, -9356.10659082084, -0.013268835868812123,
              -11.919371643223732, 2.326071473529806),
    TestValue(1.e-7, 856, 94256, -9.425599813038514e11, 9.42559999143e18,
              20.820179264456563, -9.99999999091835e6),
    TestValue(0.001, 856, 94256, -9.42451971131508e7, 9.4255143e10,
              11.60983889248038, -999.9909183500255),
    TestValue(3, 856, 94256, -27477.236822949908, 10187.222222222223,
              3.6034713248301333, -0.3242516833587959),
    TestValue(961, 856, 94256, -1100.994658834864, -0.7897178299139922,
              -2.16589079547205, 0.008041067248210687),
    TestValue(64587, 856, 94256, -4610.452335260269, -0.013246328752106911,
              -6.3736848176311955, 0.009066166982604068),
    TestValue(1.e-7, 34251, 2.5e-6, -213134.68883507163, -3.4227e11,
              -7.2225806172912765, 1.36904e10),
    TestValue(0.001, 34251, 2.5e-6, -528582.2697559998, -3.42519975e7,
              -16.43292098926746, 1.3700399e10),
    TestValue(3, 34251, 2.5e-6, -802816.3691839895, -11417.333333055556,
              -24.439288556917706, 1.3700399999666666e10),
    TestValue(961, 34251, 2.5e-6, -1.0004285605277491e6, -35.64203954214089,
              -30.20865067721989, 1.3700399999998959e10),
    TestValue(64587, 34251, 2.5e-6, -1.1445539213747415e6, -0.5303234397014873,
              -34.416444699379035, 1.3700399999999985e10),
    TestValue(1.e-7, 34251, 0.031, -200278.54089125537, 2.75748e12,
              2.2028711343018514, -8.895129032258065e6),
    TestValue(0.001, 34251, 0.031, -205782.1193121836, -3.4221e7,
              -7.007469237674332, 1.1038709677419355e6),
    TestValue(3, 34251, 0.031, -479985.2315726732, -11417.32988888889,
              -15.013836805324578, 1.1048706344086023e6),
    TestValue(961, 34251, 0.031, -677597.4126161883, -35.64203950857642,
              -20.783198925626763, 1.1048709667013527e6),
    TestValue(64587, 34251, 0.031, -821722.7734314053, -0.5303234396940565,
              -24.990992947785905, 1.1048709677264525e6),
    TestValue(1.e-7, 34251, 2.34, -2.314217999108982e7, 2.3365748e14,
              6.5267901381684545, -9.98536282051282e6),
    TestValue(0.001, 34251, 2.34, -59992.56951074866, -3.1912e7,
              -2.683550233807731, 13637.179487179486),
    TestValue(3, 34251, 2.34, -331887.4514379049, -11417.073333333334,
              -10.689917801457977, 14636.846153846154),
    TestValue(961, 34251, 2.34, -529498.8652174588, -35.64203700836256,
              -16.45927992176016, 14637.178446596761),
    TestValue(64587, 34251, 2.34, -673624.2236657206, -0.530323439140536,
              -20.667073943919306, 14637.179471696496),
    TestValue(1.e-7, 34251, 368, -3.679568940761856e9, 3.679965748e16,
              11.584722146967774, -9.999906926630436e6),
    TestValue(0.001, 34251, 368, -252413.3402773631, 3.33748e8,
              2.3743817749915905, -906.9266304347826),
    TestValue(3, 34251, 368, -158770.10887118604, -11376.444444444445,
              -5.631985792658656, 92.74003623188406),
    TestValue(961, 34251, 368, -356260.016483553, -35.6416410671766,
              -11.401347912960839, 93.07232898249106),
    TestValue(64587, 34251, 368, -500385.0000938459, -0.5303233514834095,
              -15.609141935119984, 93.07335408222546),
    TestValue(1.e-7, 34251, 94256, -9.425593789954424e11, 9.42559965748e18,
              17.13040897256031, -9.999999636617297e6),
    TestValue(0.001, 34251, 94256, -9.395046802081399e7, 9.4221748e10,
              7.920068600584127, -999.6366172975726),
    TestValue(3, 34251, 94256, -120.78940781604615, -944.4444444444445,
              -0.08629896706611895, 0.030049369094098343),
    TestValue(961, 34251, 94256, -166412.3952511924, -35.539977975595576,
              -5.855661087368302, 0.3623421197011049),
    TestValue(64587, 34251, 94256, -310441.1342976225, -0.5303008443667043,
              -10.063455109527448, 0.3633672194354983),
};

}  // namespace inv_gamma_test_internal

TEST(ProbDistributionsInvGamma, derivativesPrecomputed) {
  using inv_gamma_test_internal::TestValue;
  using inv_gamma_test_internal::testValues;
  using stan::math::inv_gamma_lpdf;
  using stan::math::is_nan;
  using stan::math::value_of;
  using stan::math::var;

  for (TestValue t : testValues) {
    var y(t.y);
    var alpha(t.alpha);
    var beta(t.beta);
    var val = inv_gamma_lpdf(y, alpha, beta);

    std::vector<var> x;
    x.push_back(y);
    x.push_back(alpha);
    x.push_back(beta);

    std::vector<double> gradients;
    val.grad(x, gradients);

    for (int i = 0; i < 3; ++i) {
      EXPECT_FALSE(is_nan(gradients[i]));
    }

    auto tolerance = [](double x) { return std::max(fabs(x * 1e-8), 1e-14); };

    EXPECT_NEAR(value_of(val), t.value, tolerance(t.value))
        << "value y = " << t.y << ", alpha = " << t.alpha
        << ", beta = " << t.beta;
    EXPECT_NEAR(gradients[0], t.grad_y, tolerance(t.grad_y))
        << "grad_y y = " << t.y << ", alpha = " << t.alpha
        << ", beta = " << t.beta;
    EXPECT_NEAR(gradients[1], t.grad_alpha, tolerance(t.grad_alpha))
        << "grad_alpha y = " << t.y << ", alpha = " << t.alpha
        << ", beta = " << t.beta;
    EXPECT_NEAR(gradients[2], t.grad_beta, tolerance(t.grad_beta))
        << "grad_beta y = " << t.y << ", alpha = " << t.alpha
        << ", beta = " << t.beta;
  }
}
