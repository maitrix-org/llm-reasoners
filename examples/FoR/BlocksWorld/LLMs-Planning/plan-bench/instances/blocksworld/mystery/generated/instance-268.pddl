(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects g d c)
(:init 
(harmony)
(planet g)
(planet d)
(planet c)
(province g)
(province d)
(province c)
)
(:goal
(and
(craves g d)
(craves d c)
)))