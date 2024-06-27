

(define (problem BW-rand-4)
(:domain blocksworld-4ops)
(:objects a b c d )
(:init
(handempty)
(ontable a)
(on b d)
(ontable c)
(on d c)
(clear a)
(clear b)
)
(:goal
(and
(on a d)
(on c a))
)
)


