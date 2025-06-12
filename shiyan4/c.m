m = zeros(6,7);
f = zeros(6,7);
d = 100;
phy0 = 50;
a = 6371;
omega = 7.292 * 10^-5;
le = 11888.45;

l_ref = (2 + sqrt(3)) * a / 2 * cosd(phy0) / (1 + sind(phy0));
