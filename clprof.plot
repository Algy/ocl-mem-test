# horizontal timelines / time bars with gnuplot ; manually rotate result image at end
# size specification controls landscape vs. portrait, this is for A4 paper size
set terminal pdf size 21cm,29.7cm
set output 'clresult.pdf'

# margins get confused so set explicitly
set lmargin at screen 0.04
set bmargin at screen 0.07
set tmargin at screen 0.97

# rotation will swap Y and X axis

# input Y data as date values
# set ydata time
# set timefmt "%Y-%m-%d"

# y coordinates now specified in time values
# set yrange ['2011-03-01':'2014-01-01']

# normal Y axis labels end up on top of graph, don't want that
unset ytics

# format y2 axis for time scale, this will appear along bottom of graph
# set y2data time
# set format y2 "%b %Y"
set y2tics font "Courier, 8"
set y2tics rotate by 90 

# y2tics 'incr' measured in seconds for y2data time, this is 4 months = 4*30*24*60*60
# set y2tics '2011-05-12',10368000,'2013-12-01'

set xrange [-1:28]
set xtics font "Courier, 8"
set xtics rotate by 90 offset 0,-5 out nomirror

# cannot rotate key (dataset label), so must create manually
unset key
# set label 'elapsed project time' at 0,'2013-06-15' rotate font "Courier, 8"
# set object 1 rect from -0.1,'2013-11-01' to 0,'2013-12-01' fillstyle solid noborder fillcolor rgb "red"

# note duplication of date columns 4 and 5 so don't get whiskers
plot "cldat.dat" using 1:4:4:5:5:xticlabels(2) with candlesticks fillcolor rgb "red"
