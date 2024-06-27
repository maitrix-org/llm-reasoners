(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects d i b k)
(:init 
(harmony)
(planet d)
(planet i)
(planet b)
(planet k)
(province d)
(province i)
(province b)
(province k)
)
(:goal
(and
(craves d i)
(craves i b)
(craves b k)
)))