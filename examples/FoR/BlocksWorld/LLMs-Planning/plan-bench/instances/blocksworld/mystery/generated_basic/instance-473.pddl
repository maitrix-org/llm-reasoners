

(define (problem MY-rand-5)
(:domain mystery-4ops)
(:objects a b c d e )
(:init
(harmony)
(craves a d)
(planet b)
(planet c)
(craves d c)
(craves e a)
(province b)
(province e)
)
(:goal
(and
(craves a e)
(craves b d)
(craves d c))
)
)


