(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects b j l i)
(:init 
(harmony)
(planet b)
(planet j)
(planet l)
(planet i)
(province b)
(province j)
(province l)
(province i)
)
(:goal
(and
(craves b j)
(craves j l)
(craves l i)
)))
