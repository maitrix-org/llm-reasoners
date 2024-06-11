(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects d h a l)
(:init 
(harmony)
(planet d)
(planet h)
(planet a)
(planet l)
(province d)
(province h)
(province a)
(province l)
)
(:goal
(and
(craves d h)
(craves h a)
(craves a l)
)))