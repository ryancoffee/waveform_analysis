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
# set terminal qt 10 font "Sans,9"
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
unset arrow
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
set xtics  norangelimit autofreq 
set ytics border in scale 1,0.5 mirror norotate  autojustify
set ytics  norangelimit autofreq 
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
set xlabel "Time-of-Flight [ns]" 
set xlabel  font "" textcolor lt -1 norotate
set x2label "" 
set x2label  font "" textcolor lt -1 norotate
set xrange [ 104.048 : 104.876 ] noreverse writeback
set x2range [ 86.7107 : 87.4007 ] noreverse writeback
set ylabel "" 
set ylabel  font "" textcolor lt -1 rotate
set y2label "" 
set y2label  font "" textcolor lt -1 rotate
set yrange [ -1.00000 : 1.00000 ] noreverse writeback
set y2range [ -5.26398 : 1.71220 ] noreverse writeback
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
file(s) = sprintf('waveforms/08_20_2020/backAndTubeAt_50V_350_2150_2550/frontAt_50V/2020_08_20_18_45_50%s',s)
GNUTERM = "qt"
file = "waveforms/08_20_2020/backAndTubeAt_100V_400_2200_2600/frontAt_100V/2020_08_20_19_45_07.zeroCrossings_histlogtimes_4.0_0.010histthresh"
x = 0.0
sigfile = "waveforms/08_14_2020/walkwayToF_70V_NoAmp_unlessNoted/370V_2170V_2570V/2020_08_14_16_48_57.samplesig"
ddfile = "waveforms/08_14_2020/walkwayToF_70V_NoAmp_unlessNoted/370V_2170V_2570V/2020_08_14_16_48_57.0.ddback"
dfile = "waveforms/08_14_2020/walkwayToF_70V_NoAmp_unlessNoted/370V_2170V_2570V/2020_08_14_16_48_57.0.dback"
yfile = "waveforms/08_14_2020/walkwayToF_70V_NoAmp_unlessNoted/370V_2170V_2570V/2020_08_14_16_48_57.0.dlogic"
back = "waveforms/08_14_2020/walkwayToF_70V_NoAmp_unlessNoted/370V_2170V_2570V/2020_08_14_16_48_57.0.back"
file2 = "waveforms/08_14_2020/walkwayToF_70V_NoAmp_unlessNoted/370V_2370V_2770V/2020_08_14_17_16_10.zeroCrossings_histlogtimes_4.0_1.0histthresh"
file3 = "waveforms/08_14_2020/walkwayToF_70V_NoAmp_unlessNoted/370V_1970V_2370V/2020_08_14_17_49_15.zeroCrossings_histlogtimes_4.0_0.001histthresh"
file50 = "waveforms/08_20_2020/backAndTubeAt_50V_350_2150_2550/frontAt_50V/2020_08_20_18_45_50.zeroCrossings_histlogtimes_4.0_0.010histthresh"
file100 = "waveforms/08_20_2020/backAndTubeAt_100V_400_2200_2600/frontAt_100V/2020_08_20_19_45_07.zeroCrossings_histlogtimes_4.0_0.010histthresh"
file150 = "waveforms/08_20_2020/backAndTubeAt_150V_450_2250_2650/frontAt_150V/2020_08_20_20_39_07.zeroCrossings_histlogtimes_4.0_0.010histthresh"
file70 = "waveforms/08_14_2020/walkwayToF_70V_NoAmp_unlessNoted/370V_2370V_2770V/2020_08_14_17_16_10.zeroCrossings_histlogtimes_4.0_1.0histthresh"
## Last datafile plotted: "waveforms/08_20_2020/backAndTubeAt_50V_350_2150_2550/frontAt_50V/2020_08_20_18_45_50.0.dlogic"
plot file('.samplesig') u 1:(7*$6),file('.0.back') u 1:6,file('.0.dback') u 1:6,file('.0.ddback') u 1:6,file('.0.dlogic') u (exp($1)):($6*10) lc -1 w points,file('.0.dlogic') u (exp($1)):($6*10) every 6 lt 7 lc rgb 'red' w points
#    EOF
