(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects b d i)
(:init 
(harmony)
(planet b)
(planet d)
(planet i)
(province b)
(province d)
(province i)
)
(:goal
(and
(craves b d)
(craves d i)
)))