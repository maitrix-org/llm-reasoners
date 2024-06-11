(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects b l d i)
(:init 
(harmony)
(planet b)
(planet l)
(planet d)
(planet i)
(province b)
(province l)
(province d)
(province i)
)
(:goal
(and
(craves b l)
(craves l d)
(craves d i)
)))