(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects l d b c)
(:init 
(harmony)
(planet l)
(planet d)
(planet b)
(planet c)
(province l)
(province d)
(province b)
(province c)
)
(:goal
(and
(craves l d)
(craves d b)
(craves b c)
)))