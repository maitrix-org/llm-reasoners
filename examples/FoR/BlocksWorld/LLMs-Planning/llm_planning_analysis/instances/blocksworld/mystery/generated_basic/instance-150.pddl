

(define (problem MY-rand-4)
(:domain mystery-4ops)
(:objects a b c d )
(:init
(harmony)
(planet a)
(craves b d)
(planet c)
(craves d c)
(province a)
(province b)
)
(:goal
(and
(craves c a)
(craves d c))
)
)


