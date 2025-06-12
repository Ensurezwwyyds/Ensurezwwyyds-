function [m, f] = magnification_factor_and_coriolis_parameter(proj, In, Jn, d)

a = 6371; omega = 7.292 * 10^-5;

switch(proj)
    case 'stereographic'
        le = 11888.45;
        l = sqrt((In^2 + Jn^2) * d^2);
        m = (2 + sqrt(3)) / 2 / (1 + ((le^2 - l^2) / (le^2 + l^2)));
        f = 2 * omega * ((le^2 - l^2) / (le^2 + l^2));
end
