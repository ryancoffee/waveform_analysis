#!/usr/bin/gnuplot -persist
#
#    
#    	G N U P L O T
#    	Version 5.2 patchlevel 8    last modified 2019-12-01 
#    
#    	Copyright (C) 1986-1993, 1998, 2004, 2007-2019
#    	Thomas Williams, Colin Kelley and many others
#    
#    	gnuplot home:     http://www.gnuplot.info
#    	faq, bugs, etc:   type "help FAQ"
#    	immediate help:   type "help"  (plot window: hit 'h')
# set terminal qt 1 font "Sans,9"
# set output
unset clip points
set clip one
unset clip two
set errorbars front 1.000000 
set border 31 front lt black linewidth 1.000 dashtype solid
set zdata 
set ydata 
set xdata 
set y2data 
set x2data 
set boxwidth
set style fill  empty border
set style rectangle back fc  bgnd fillstyle   solid 1.00 border lt -1
set style circle radius graph 0.02 
set style ellipse size graph 0.05, 0.03 angle 0 units xy
set dummy x, y
set format x "% h" 
set format y "% h" 
set format x2 "% h" 
set format y2 "% h" 
set format z "% h" 
set format cb "% h" 
set format r "% h" 
set ttics format "% h"
set timefmt "%d/%m/%y,%H:%M"
set angles radians
set tics back
unset grid
unset raxis
set theta counterclockwise right
set style parallel front  lt black linewidth 2.000 dashtype solid
set key title "" center
set key fixed right top vertical Right noreverse enhanced autotitle nobox
set key noinvert samplen 4 spacing 1 width 0 height 0 
set key maxcolumns 0 maxrows 0
set key noopaque
unset label
set label 1 "high frequency\nedge of signal" at 2.00000, 2.00000, 0.00000 center norotate back nopoint
set label 2 "normalize to\nlow frequency\nsignal power" at 0.500000, 200.000, 0.00000 center norotate back nopoint
unset arrow
set arrow 1 from 2.00000, 3.00000, 0.00000 to 2.00000, 10.0000, 0.00000 head back filled linewidth 1.000 dashtype solid
set arrow 2 from 0.500000, 300.000, 0.00000 to 0.500000, 1000.00, 0.00000 head back filled linewidth 1.000 dashtype solid
set style increment default
unset style line
unset style arrow
set style histogram clustered gap 2 title textcolor lt -1
unset object
set style textbox transparent margins  1.0,  1.0 border  lt -1 linewidth  1.0
set offsets 0, 0, 0, 0
set pointsize 1
set pointintervalbox 1
set encoding default
unset polar
unset parametric
unset decimalsign
unset micro
unset minussign
set view 60, 30, 1, 1
set view azimuth 0
set rgbmax 255
set samples 100, 100
set isosamples 10, 10
set surface 
unset contour
set cntrlabel  format '%8.3g' font '' start 5 interval 20
set mapping cartesian
set datafile separator whitespace
unset hidden3d
set cntrparam order 4
set cntrparam linear
set cntrparam levels 5
set cntrparam levels auto
set cntrparam firstlinetype 0 unsorted
set cntrparam points 5
set size ratio 0 1,1
set origin 0,0
set style data histeps
set style function lines
unset xzeroaxis
unset yzeroaxis
unset zzeroaxis
unset x2zeroaxis
unset y2zeroaxis
set xyplane relative 0.5
set tics scale  1, 0.5, 1, 1, 1
set mxtics default
set mytics default
set mztics default
set mx2tics default
set my2tics default
set mcbtics default
set mrtics default
set nomttics
set xtics border in scale 1,0.5 mirror norotate  autojustify
set xtics  norangelimit 0.5
set ytics border in scale 1,0.5 mirror norotate  autojustify
set ytics  norangelimit logscale autofreq 
set ztics border in scale 1,0.5 nomirror norotate  autojustify
set ztics  norangelimit autofreq 
unset x2tics
unset y2tics
set cbtics border in scale 1,0.5 mirror norotate  autojustify
set cbtics  norangelimit autofreq 
set rtics axis in scale 1,0.5 nomirror norotate  autojustify
set rtics  norangelimit autofreq 
unset ttics
set title "" 
set title  font "" textcolor lt -1 norotate
set timestamp bottom 
set timestamp "" 
set timestamp  font "" textcolor lt -1 norotate
set trange [ * : * ] noreverse nowriteback
set urange [ * : * ] noreverse nowriteback
set vrange [ * : * ] noreverse nowriteback
set xlabel "frequency [GHz]" 
set xlabel  font "" textcolor lt -1 norotate
set x2label "" 
set x2label  font "" textcolor lt -1 norotate
set xrange [ 0.00000 : 4.00000 ] noreverse writeback
set x2range [ * : * ] noreverse writeback
set ylabel "power [arb.]" 
set ylabel  font "" textcolor lt -1 rotate
set y2label "" 
set y2label  font "" textcolor lt -1 rotate
set yrange [ 1.00000 : 10000.0 ] noreverse writeback
set y2range [ * : * ] noreverse writeback
set zlabel "" 
set zlabel  font "" textcolor lt -1 norotate
set zrange [ * : * ] noreverse writeback
set cblabel "" 
set cblabel  font "" textcolor lt -1 rotate
set cbrange [ * : * ] noreverse writeback
set rlabel "" 
set rlabel  font "" textcolor lt -1 norotate
set rrange [ * : * ] noreverse writeback
unset logscale
set logscale y 10
unset jitter
set zero 1e-08
set lmargin  -1
set bmargin  -1
set rmargin  -1
set tmargin  -1
set locale "en_US.UTF-8"
set pm3d explicit at s
set pm3d scansautomatic
set pm3d interpolate 1,1 flush begin noftriangles noborder corners2color mean
set pm3d nolighting
set palette positive nops_allcF maxcolors 0 gamma 1.5 color model RGB 
set palette rgbformulae 7, 5, 15
set colorbox default
set colorbox vertical origin screen 0.9, 0.2 size screen 0.05, 0.6 front  noinvert bdefault
set style boxplot candles range  1.50 outliers pt 7 separation 1 labels auto unsorted
set loadpath 
set fontpath 
set psdir
set fit brief errorvariables nocovariancevariables errorscaling prescale nowrap v5
spectname(i) = sprintf('/nvme/hpl-CookieBox_testing/09_04_2020/30V_noAmp/330V_%iV_2180V/2020_09_04_16_27_03_powerspect.out',i)
GNUTERM = "qt"
spectname1780 = "/nvme/hpl-CookieBox_testing/09_04_2020/30V_noAmp/330V_1780V_2180V/2020_09_04_16_27_03_powerspect.out"
spectname1830 = "/nvme/hpl-CookieBox_testing/09_04_2020/30V_noAmp/330V_1830V_2230V/2020_09_04_15_34_34_powerspect.out"
spectname1880 = "/nvme/hpl-CookieBox_testing/09_04_2020/30V_noAmp/330V_1880V_2280V/2020_09_04_15_47_12_powerspect.out"
spectname1930 = "/nvme/hpl-CookieBox_testing/09_04_2020/30V_noAmp/330V_1930V_2330V/2020_09_04_15_59_06_powerspect.out"
spectname1980 = "/nvme/hpl-CookieBox_testing/09_04_2020/30V_noAmp/330V_1980V_2380V/2020_09_04_16_12_37_powerspect.out"
ampspectname1630 = "/nvme/hpl-CookieBox_testing/09_04_2020/30V_withAmp/330V_1630V_2030V/2020_09_04_17_25_23_powerspect.out"
ampspectname1680 = "/nvme/hpl-CookieBox_testing/09_04_2020/30V_withAmp/330V_1680V_2080V/2020_09_04_17_11_54_powerspect.out"
ampspectname1730 = "/nvme/hpl-CookieBox_testing/09_04_2020/30V_withAmp/330V_1730V_2130V/2020_09_04_16_58_45_powerspect.out"
ampspectname1780 = "/nvme/hpl-CookieBox_testing/09_04_2020/30V_withAmp/330V_1780V_2180V/2020_09_04_16_45_12_powerspect.out"
ampspectname1830 = "/nvme/hpl-CookieBox_testing/09_04_2020/30V_withAmp/330V_1830V_2230V/2020_09_04_17_42_08_powerspect.out"
## Last datafile plotted: "/nvme/hpl-CookieBox_testing/09_04_2020/30V_noAmp/330V_1930V_2330V/2020_09_04_15_59_06_powerspect.out"
set term png size 600,600
set output './figs/plotting.spectVgain.amps.png'
plot 	ampspectname1630 title '1300 V',\
	ampspectname1680 u 1:($2/3) title '1350 V',\
	ampspectname1730 u 1:($2/15) title '1400 V',\
	ampspectname1780 u 1:($2/60) title '1450 V',\
	ampspectname1830 u 1:($2/200) title '1500 V',\
	spectname1930 u 1:($2/35) title '1600 V, no Amp',\
	
set term png size 600,600
set arrow 3 filled from .15,7e3 to .15,4e3
set label 3 left at .17,7.5e3 "greater than 1600V adds\nonly low frequency power"
set output './figs/plotting.spectVgain.straight.png'
plot 	spectname1780 u 1:($2) title '1450 V',\
	spectname1830 u 1:($2/1.75) title '1500 V',\
	spectname1880 u 1:($2/8) title '1550 V',\
	spectname1930 u 1:($2/25) title '1600 V',\
	spectname1980 u 1:($2/70) title '1650 V'
#    EOF
