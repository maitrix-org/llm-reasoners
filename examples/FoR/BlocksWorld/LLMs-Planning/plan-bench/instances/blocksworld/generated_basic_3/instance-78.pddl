

(define (problem BW-rand-3)
(:domain blocksworld-4ops)
(:objects a b c )
(:init
(handempty)
(on a b)
(ontable b)
(on c a)
(clear c)
)
(:goal
(and
(on b c))
)
)


