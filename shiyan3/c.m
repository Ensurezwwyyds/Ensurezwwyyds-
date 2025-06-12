switch(proj)
    case 'mercator'
        m=sqrt((a * cosd(22.5))^2 + (Jn * d)^2) / a;
        f=2 * omega * sin(Jn * d / sqrt((a * cosd(22.5))^2 + (Jn * d)^2));
end
