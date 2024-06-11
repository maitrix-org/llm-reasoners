(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects d l f a)
(:init 
(harmony)
(planet d)
(planet l)
(planet f)
(planet a)
(province d)
(province l)
(province f)
(province a)
)
(:goal
(and
(craves d l)
(craves l f)
(craves f a)
)))