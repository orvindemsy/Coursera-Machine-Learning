%Author : Orvin Demsy
%Date created : May, 25th 2019
%I'm trying to understand why the Z in surf(x,y,z) needs to be transposed
%before executed
%I'm trying to understand what does this sentence mean 'Because of the way meshgrids work in the surf command, we need to
%transpose J_vals before calling surf, or else the axes will be flipped'

%Result : I kinda found my aha moment
%I think you have to do both with the one resulting in square matrix and
%one in rectangular matrix (i.e. the one which has even row and colum, and
%the one that doesn't)

a = linspace(1,10,10);
b = linspace(-5,4,10);
z3 = zeros(length(a), length(b));

for i = 1:length(a)
    for j = 1:length(b)
    z3(i,j) = squareTwoNumbers(a(i), b(j))
    %z3(i,j) = sumTwoNumbers(a(i), b(j)) %The reason this one produced 
    %same graph for transposed and non-transposed is because the funtion
    %happened to be linear (check the correctness of data sets).
    end
end
figure(1);
surf(a, b, z3);
xlabel('x');
ylabel('y');
zlabel('z');
title('\fontsize{13} This one is not transposed');

figure(2);
surf(a, b, z3');
xlabel('x');
ylabel('y');
zlabel('z');
title('\fontsize{13} This one is transposed');

%{
a = linspace(1,10,10);
b = linspace(-5,5,11);
b_trans = b';
a_trans = a';

z1 = b_trans* a;
z2 = a_trans* b;

surf(a, b, z1);
xlabel('x');
ylabel('y');
zlabel('z');
%}