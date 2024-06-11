

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(craves a e)
(planet b)
(craves c b)
(craves d a)
(planet e)
(province c)
(province d)
)
(:goal
(and
(craves b e)
(craves c d)
(craves d a))
)
)


