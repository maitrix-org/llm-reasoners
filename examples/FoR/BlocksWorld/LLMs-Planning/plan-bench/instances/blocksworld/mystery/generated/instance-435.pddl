(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects g d f b)
(:init 
(harmony)
(planet g)
(planet d)
(planet f)
(planet b)
(province g)
(province d)
(province f)
(province b)
)
(:goal
(and
(craves g d)
(craves d f)
(craves f b)
)))