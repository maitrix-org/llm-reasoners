(define (problem BW-generalization-4)
(:domain mystery-4ops)(:objects b l e g)
(:init 
(harmony)
(planet b)
(planet l)
(planet e)
(planet g)
(province b)
(province l)
(province e)
(province g)
)
(:goal
(and
(craves b l)
(craves l e)
(craves e g)
)))